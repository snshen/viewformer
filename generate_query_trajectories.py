import os
import re
import numpy as np
from PIL import Image

from viewformer.utils.geometry import look_at_to_cameras
from viewformer.data._common import ArchiveStore

class QueryTrajectories:

    def __init__(self, fname) -> None:
        self.fname = fname
        self.data = []
        par_dir, archivename = os.path.split(fname)
        par_dir = os.path.join(os.path.dirname(par_dir), 'GroundTruth_HD1-HD6')
        with ArchiveStore(os.path.join(par_dir, archivename)) as gt_archive:
            subdirs = [re.match(r'^.*(\d+_\d+)$', x) for x in gt_archive.ls('')]
            subdir_postfixes = [x.group(1) for x in subdirs if x is not None]
            subdirs = [f'original_{x}/' for x in subdir_postfixes]
            for subdir, postfix in zip(subdirs, subdir_postfixes):
                with gt_archive.open(f'velocity_angular_{postfix}/cam0.render', 'r') as f:
                    for pose_id, pose_data in self._parse_cam(f):
                        self.data.append((subdir, pose_id, pose_data))

    def get_traj(self, start_frame, traj_len, frame_skip):
        images = []
        cameras = []
        output = dict()
        with ArchiveStore(self.fname) as archive:
            for i in range(start_frame, start_frame+traj_len*frame_skip, frame_skip):
                if i >= len(self.data):
                    print("START FRAME OUT OF BOUNDS. MUST BE < ", len(self.data))
                    return output
                subdir, pose_id, pose_data = self.data[i]
                image = np.array(Image.open(archive.open(f'{subdir}cam0/data/{pose_id}.png', 'rb')).convert('RGB'))
                images.append(image)
                cameras.append(pose_data)
        
        cameras = np.stack(cameras, 0)
        cameras = self._convert_poses(cameras)
        output['cameras'] = cameras
        output['frames'] = np.stack(images, 0)
        return output

    def _parse_cam(self, file):
        last_id = None
        for line in file:
            line = line.rstrip('\n\r')
            vals = line.split()
            if vals[0].isnumeric():
                if last_id != vals[0]:
                    yield vals[0], np.array([float(x) for x in vals[1:]], dtype='float32')
                last_id = vals[0]

    def _rotate_system(self, pos):
        x, y, z = np.moveaxis(pos, -1, 0)
        return np.stack((y, -z, -x), -1)

    def _convert_poses(self, poses):
        # first three elements, eye and next three, lookAt and the last there, up direction
        eye = self._rotate_system(poses[..., 0:3])
        lookat = self._rotate_system(poses[..., 3:6])
        up = self._rotate_system(poses[..., 6:9])
        return look_at_to_cameras(eye, lookat, up)


if __name__ == '__main__':
    qt = QueryTrajectories('/home/ec2-user/novel-cross-view-generation/viewformer/data/HD6/3FO4JTCNVCGA')
    output = qt.get_traj(0, 10, 10)
    print(output['cameras'])