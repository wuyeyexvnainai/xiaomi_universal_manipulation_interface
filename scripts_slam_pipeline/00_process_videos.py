"""
python scripts_slam_pipeline/00_process_videos.py data_workspace/toss_objects/20231113
"""
# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import pathlib
import click
import shutil
from exiftool import ExifToolHelper
from umi.common.timecode_util import mp4_get_start_datetime

# %%
@click.command(help='Session directories. Assumming mp4 videos are in <session_dir>/raw_videos')
@click.argument('session_dir', nargs=-1)
def main(session_dir):
    for session in session_dir:
        # expanduser:将～替换为当前用户的主目录路径
        # absolute()：替换为绝对路径
        session = pathlib.Path(os.path.expanduser(session)).absolute()
        # hardcode subdirs
        input_dir = session.joinpath('raw_videos')
        output_dir = session.joinpath('demos')
        
        # create raw_videos if don't exist
        if not input_dir.is_dir():
            input_dir.mkdir()
            print(f"{input_dir.name} subdir don't exits! Creating one and moving all mp4 videos inside.")
            # 找出session文件夹里面所有后缀为MP4/mp4的文件地址
            for mp4_path in list(session.glob('**/*.MP4')) + list(session.glob('**/*.mp4')):
                # 。。。.name求得该地址下文件的名字
                out_path = input_dir.joinpath(mp4_path.name)
                # 将MP4文件移动到out_path目录下
                shutil.move(mp4_path, out_path)
        
        # create mapping video if don't exist：挑选出内存最大的文件
        mapping_vid_path = input_dir.joinpath('mapping.mp4')
        # .is_symlink()判断是否有符号连接
        if (not mapping_vid_path.exists()) and not(mapping_vid_path.is_symlink()):
            max_size = -1
            max_path = None
            for mp4_path in list(input_dir.glob('**/*.MP4')) + list(input_dir.glob('**/*.mp4')):
                size = mp4_path.stat().st_size
                if size > max_size:
                    max_size = size
                    max_path = mp4_path
            shutil.move(max_path, mapping_vid_path)
            print(f"raw_videos/mapping.mp4 don't exist! Renaming largest file {max_path.name}.")
        
        # create gripper calibration video if don't exist
        gripper_cal_dir = input_dir.joinpath('gripper_calibration')
        if not gripper_cal_dir.is_dir():
            gripper_cal_dir.mkdir()
            print("raw_videos/gripper_calibration don't exist! Creating one with the first video of each camera serial.")
            # 建立字典
            serial_start_dict = dict()
            serial_path_dict = dict()
            with ExifToolHelper() as et:
                for mp4_path in list(input_dir.glob('**/*.MP4')) + list(input_dir.glob('**/*.mp4')):
                    # 如果文件以MP4的文件名
                    if mp4_path.name.startswith('map'):
                        continue
                    
                    start_date = mp4_get_start_datetime(str(mp4_path))
                    meta = list(et.get_metadata(str(mp4_path)))[0]
                    # 提取相机的序列号
                    cam_serial = meta['QuickTime:CameraSerialNumber']
                    
                    if cam_serial in serial_start_dict:
                        if start_date < serial_start_dict[cam_serial]:
                            serial_start_dict[cam_serial] = start_date
                            serial_path_dict[cam_serial] = mp4_path
                    else:
                        serial_start_dict[cam_serial] = start_date
                        serial_path_dict[cam_serial] = mp4_path
            
            for serial, path in serial_path_dict.items():
                print(f"Selected {path.name} for camera serial {serial}")
                out_path = gripper_cal_dir.joinpath(path.name)
                shutil.move(path, out_path)

        # look for mp4 video in all subdirectories in input_dir
        input_mp4_paths = list(input_dir.glob('**/*.MP4')) + list(input_dir.glob('**/*.mp4'))
        print(f'Found {len(input_mp4_paths)} MP4 videos')

        with ExifToolHelper() as et:
            for mp4_path in input_mp4_paths:
                if mp4_path.is_symlink():
                    print(f"Skipping {mp4_path.name}, already moved.")
                    continue

                start_date = mp4_get_start_datetime(str(mp4_path))
                meta = list(et.get_metadata(str(mp4_path)))[0]
                cam_serial = meta['QuickTime:CameraSerialNumber']
                out_dname = 'demo_' + cam_serial + '_' + start_date.strftime(r"%Y.%m.%d_%H.%M.%S.%f")

                # special folders
                if mp4_path.name.startswith('mapping'):
                    out_dname = "mapping"
                elif mp4_path.name.startswith('gripper_cal') or mp4_path.parent.name.startswith('gripper_cal'):
                    out_dname = "gripper_calibration_" + cam_serial + '_' + start_date.strftime(r"%Y.%m.%d_%H.%M.%S.%f")
                
                # create directory
                this_out_dir = output_dir.joinpath(out_dname)
                this_out_dir.mkdir(parents=True, exist_ok=True)
                
                # move videos
                vfname = 'raw_video.mp4'
                out_video_path = this_out_dir.joinpath(vfname)
                shutil.move(mp4_path, out_video_path)

                # create symlink back from original location
                # relative_to's walk_up argument is not avaliable until python 3.12
                # 求mp4与session之间的相对位置，即在mp4路径下面cd dots可以进入session路径
                dots = os.path.join(*['..'] * len(mp4_path.parent.relative_to(session).parts))
                # 求out_video_path与session之间的相对路径
                rel_path = str(out_video_path.relative_to(session))
                # 先回退到session路径，然后再向前进入到out_video_path路径
                symlink_path = os.path.join(dots, rel_path)
                # 建立快捷方式   
                mp4_path.symlink_to(symlink_path)

# %%
if __name__ == '__main__':
    # 如果在命令行没有传递给python脚本其他参数，触发帮助信息显示
    if len(sys.argv) == 1:
        main.main(['--help'])
    else:
        main()
