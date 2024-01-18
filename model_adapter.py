import json
import dtlpy as dl
import os
import torch
import torch.nn.functional as F
from model import GroundEstimatorNet
from utils.utils import segment_cloud
from utils.point_cloud_ops import points_to_voxel
import yaml
import numpy as np
import open3d as o3d
import logging
import shutil
import time
import tqdm
import pathlib
import urllib.request

logger = logging.getLogger('ModelAdapter')


class ConfigClass:
    def __init__(self, **entries):
        self.__dict__.update(entries)


class ModelAdapter(dl.BaseModelAdapter):
    def __init__(self, model_entity):
        logger.info("Starting model initialization")
        time_started = time.time()
        self.pcd_mapping = dict()
        self.checkpoint = ''
        self.cfg = None
        self.start_epoch = 0
        super(ModelAdapter, self).__init__(model_entity)
        logger.info(f"Model initialization took {time.time() - time_started} seconds")

    def load(self, local_path, **kwargs):
        config = pathlib.Path(__file__).parent / pathlib.Path('config/config_kittiSem.yaml')
        self.checkpoint = "artifacts/checkpoint.pth.tar"
        checkpoint_url = r'https://storage.googleapis.com/model-mgmt-snapshots/GndNet/checkpoint.pth.tar'
        if not os.path.isfile(self.checkpoint):
            os.makedirs(os.path.dirname(self.checkpoint), exist_ok=True)
            urllib.request.urlretrieve(checkpoint_url, self.checkpoint)
            if not os.path.isfile(self.checkpoint):
                raise FileNotFoundError(f"Could not find checkpoint file at {self.checkpoint}")
        if os.path.isfile(config):
            with open(config) as f:
                config_dict = yaml.load(f, Loader=yaml.FullLoader)
        self.cfg = ConfigClass(**config_dict)
        self.cfg.batch_size = 1
        self.model = GroundEstimatorNet(self.cfg).cuda()
        if os.path.isfile(self.checkpoint):
            checkpoint = torch.load(self.checkpoint)
            self.start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])

    @staticmethod
    def _shift_cloud(cloud, height):
        cloud = np.concatenate((cloud, np.ones((cloud.shape[0], 1))), axis=1)
        cloud += np.array([0, 0, height, 0], dtype=np.float32)
        return cloud[:, :3]

    def infer_ground(self, cloud):
        # TODO: try removing shift cloud
        cloud = self._shift_cloud(cloud[:, :4], self.cfg.lidar_height)
        if cloud.shape[1] == 3:
            cloud = np.concatenate((cloud, np.zeros((cloud.shape[0], 1))), axis=1)
        voxels, coors, num_points = points_to_voxel(points=cloud,
                                                    voxel_size=self.cfg.voxel_size,
                                                    coors_range=self.cfg.pc_range,
                                                    max_points=self.cfg.max_points_voxel,
                                                    reverse_index=True,
                                                    max_voxels=self.cfg.max_voxels)
        voxels = torch.from_numpy(voxels).float().cuda()
        coors = torch.from_numpy(coors)
        coors = F.pad(coors, (1, 0), 'constant', 0).float().cuda()
        num_points = torch.from_numpy(num_points).float().cuda()
        with torch.no_grad():
            output = self.model(voxels, coors, num_points)
        return output

    @staticmethod
    def find_pcd_path(pcd_id, local_path):
        for root, dirs, files in os.walk(local_path):
            for file in files:
                if file.endswith('.json'):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        if data.get('id') == pcd_id:
                            file_path = file_path.replace('.json', '.pcd')
                            file_path = file_path.replace('json', 'items')
                            return file_path

    def map_pcd_by_id(self, item: dl.Item, local_path):
        self.logger.info("Starting data download")
        time_started = time.time()
        filters = dl.Filters()
        filters.add(field='metadata.system.mimetype', values='*pcd')
        item.dataset.items.download(filters=filters,
                                    local_path=local_path,
                                    annotation_options=dl.ViewAnnotationOptions.JSON)
        buffer = item.download(save_locally=False)
        mapping_data = json.loads(buffer.getvalue())
        for frame_num, frame_details in enumerate(mapping_data.get('frames')):
            local_file_path = frame_details.get('lidar', dict()).get('remote_path', None)
            if local_file_path is not None and len(local_file_path) > 1:
                local_file_path = os.path.join(local_path, 'items', local_file_path[1:])
            else:
                local_file_path = self.find_pcd_path(pcd_id=frame_details.get('lidar', dict()).get('lidar_pcd_id'),
                                                     local_path=os.path.join(local_path, 'json'))
            translation = frame_details.get('translation', dict())
            self.pcd_mapping[frame_num] = {
                'translation': [translation.get('x'), translation.get('y'), translation.get('z')],
                'item_id': frame_details.get('lidar', dict()).get('lidar_pcd_id'),
                'local_path': local_file_path
            }
        logger.info(f"Downloading data took {time.time() - time_started} seconds")

    def predict(self, batch, **kwargs):
        pcd_path = os.path.join(os.getcwd(), 'pcd_path')
        os.makedirs(pcd_path, exist_ok=True)
        self.map_pcd_by_id(item=batch, local_path=pcd_path)

        ground_path = os.path.join(os.getcwd(), 'ground')
        os.makedirs(ground_path, exist_ok=True)
        progress = tqdm.tqdm(total=len(self.pcd_mapping))
        total_upload_time = 0
        for frame_num, frame_details in self.pcd_mapping.items():
            item = dl.items.get(item_id=frame_details.get('item_id'))
            full_pcd_path = frame_details.get('local_path')
            pcd = o3d.io.read_point_cloud(full_pcd_path)
            translation = np.array(frame_details.get('translation'))
            threshold = 0.1
            # TODO: If this doesn't work in the future,
            #  try to use the open3d plane segmentation for height, refer to lidar-preprocess bitbucket
            if list(translation) == [0, 0, 0]:
                # Sort Lowest 15% of points by Z value
                z_values = list(np.asarray(pcd.points)[:, 2])
                lowest_15_percent_num = int(len(z_values) * 0.15)
                sorted_list = sorted(z_values)
                lowest_15_percent = sorted_list[:lowest_15_percent_num]
                highest_z_15_percent = lowest_15_percent[-1]
                lowest_z_15_percent = lowest_15_percent[0]
                # Create a dynamic threshold for the ground segmentation
                threshold = highest_z_15_percent - lowest_z_15_percent + 0.1
                # create translation vector that will move the scene to z=0 area
                translation = [0, 0, highest_z_15_percent]
            points = np.asarray(pcd.points) - translation
            pred_gnd = self.infer_ground(points.copy())
            pred_gnd = pred_gnd.cpu().numpy()
            pred_GndSeg = segment_cloud(points=points.copy(),
                                        grid_size=np.asarray(self.cfg.grid_range),
                                        voxel_size=self.cfg.voxel_size[0],
                                        elevation_map=pred_gnd.T,
                                        threshold=threshold)

            ground = list()
            for idx, point in enumerate(pred_GndSeg):
                if point == 0:
                    ground.append(idx)

            ground_file = os.path.join(ground_path, "{}.txt".format(item.id))
            time_started = time.time()
            with open(ground_file, "w") as f1:
                f1.write(str(ground))
                uploaded_ground = item.dataset.items.upload(local_path=ground_file,
                                                            remote_name='{}_{}.{}'.format(item.id,
                                                                                          'MAP', 'txt'),
                                                            remote_path='/.dataloop/lidar_ground',
                                                            overwrite=True)
                if 'user' not in item.metadata:
                    item.metadata['user'] = dict()
                if 'lidar_ground_detection' not in item.metadata['user']:
                    item.metadata['user']['lidar_ground_detection'] = dict()
                item.metadata['user']['lidar_ground_detection']['groundMapId'] = uploaded_ground.id
                item.update()
            progress.update()
            total_upload_time += time_started - time.time()

        self.logger.info(f"Time it took for uploading ground files {total_upload_time}")
        if os.path.exists(pcd_path):
            shutil.rmtree(pcd_path)
        if os.path.exists(ground_path):
            shutil.rmtree(ground_path)

    def predict_items(self, items: list, **kwargs):
        """
        Run the predict function on the input list of items (or single) and return the items and the predictions.
        Each prediction is by the model output type (package.output_type) and model_info in the metadata
        :param items: `List[dl.Item]` list of items to predict
        """
        for item in items:
            logger.info(f"Starting prediction on scene {item.id}")
            time_started = time.time()
            self.predict(item, **kwargs)
            logger.info(f"prediction on entire lidar scene {item.id} took {time.time() - time_started} seconds")
