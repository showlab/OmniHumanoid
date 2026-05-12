# from torch.utils.data import Dataset
# import os
# from decord import VideoReader
# from torchvision import transforms
# import numpy as np
# from PIL import Image
# import torch

# class CustomDataset(Dataset):
#     def __init__(
#         self,
#         video_root,
#         video_root2,
#         first_root,
#         height=512,
#         width=512,
#         sample_n_frames=49,
#         is_one2three=False,
#         training_len=-1,
#         caption_ext=".txt",  # 文本文件后缀
#     ):
#         self.training_len = training_len
#         self.is_one2three = is_one2three

#         self.video_root = video_root
#         self.video_root2 = video_root2
#         self.caption_ext = caption_ext

#         # --- 可选首帧：目录存在且有 >=1 个可读图片才开启 ---
#         img_exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
#         if first_root and os.path.isdir(first_root):
#             first_list_all = sorted(os.listdir(first_root))
#             first_list = [x for x in first_list_all if x.lower().endswith(img_exts)]
#             if len(first_list) > 0:
#                 self.first_root = first_root
#                 self.first_paths = [os.path.join(first_root, x) for x in first_list]
#                 self.use_first = True
#             else:
#                 self.first_root = None
#                 self.first_paths = []
#                 self.use_first = False
#         else:
#             self.first_root = None
#             self.first_paths = []
#             self.use_first = False

#         print(
#             f"CustomDataset: video_root: {video_root}, "
#             f"video_root2: {video_root2}, first_root: {self.first_root or ''}"
#         )

#         video_exts = (".mp4", ".avi", ".mov", ".mkv")

#         # 视频列表（只收视频后缀）
#         video_list = sorted(
#             [x for x in os.listdir(self.video_root) if x.lower().endswith(video_exts)]
#         )
#         video_list2 = sorted(
#             [x for x in os.listdir(self.video_root2) if x.lower().endswith(video_exts)]
#         )

#         self.video_paths = [os.path.join(self.video_root, v) for v in video_list]
#         self.video_paths2 = [os.path.join(self.video_root2, v) for v in video_list2]

#         self.height = height
#         self.width = width
#         self.train_video_transforms = transforms.Compose(
#             [
#                 transforms.CenterCrop((height, width)),
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
#             ]
#         )

#         self.sample_n_frames = sample_n_frames

#         self.len_videos = len(self.video_paths)
#         self.len_videos2 = len(self.video_paths2)
#         self.len_firsts = len(self.first_paths)

#         # 两路视频必须一一对应
#         assert self.len_videos == self.len_videos2, "mismatch in first videos and third videos"

#     def __len__(self):
#         if self.training_len != -1:
#             return self.training_len
#         if self.use_first:
#             # 仅当真的有首帧时才把 len_firsts 纳入长度
#             return min(self.len_videos, self.len_videos2, self.len_firsts)
#         else:
#             return min(self.len_videos, self.len_videos2)

#     def _caption_path_for(self, video2_path: str):
#         stem, _ = os.path.splitext(video2_path)
#         return stem + self.caption_ext

#     def _load_caption(self, video2_path: str) -> str:
#         cap_path = self._caption_path_for(video2_path)
#         if os.path.exists(cap_path) and os.path.isfile(cap_path):
#             try:
#                 with open(cap_path, "r", encoding="utf-8") as f:
#                     return f.read().strip()
#             except Exception:
#                 return ""
#         return ""

#     def __getitem__(self, index):
#         # 根据是否启用首帧决定取模长度，避免越界
#         if self.use_first:
#             min_len = min(self.len_videos, self.len_videos2, self.len_firsts)
#         else:
#             min_len = min(self.len_videos, self.len_videos2)

#         # 如果真的没有数据，给出清晰报错（避免除以 0 或空取模）
#         if min_len <= 0:
#             raise RuntimeError(
#                 f"No valid samples: "
#                 f"len_videos={self.len_videos}, len_videos2={self.len_videos2}, len_firsts={self.len_firsts}."
#             )

#         index = index % min_len

#         # video A
#         video_path = self.video_paths[index]
#         video_reader = VideoReader(video_path)
#         video_length = len(video_reader)

#         # video B (GT)
#         video_path2 = self.video_paths2[index]
#         video_reader2 = VideoReader(video_path2)
#         video_length2 = len(video_reader2)

#         # 可选的首帧
#         first_frame = None
#         if self.use_first:
#             # 这里安全：index < len_firsts
#             first_frame_path = self.first_paths[index]
#             first_frame = Image.open(first_frame_path).convert("RGB")
#             first_frame = self.train_video_transforms(first_frame)

#         assert video_length == video_length2, "video lengths do not match"
#         assert self.sample_n_frames <= video_length, "sample_n_frames > video length"

#         # 采样帧（stride=2），并避免 randint 上界为负
#         stride = 1
#         available = video_length - (self.sample_n_frames - 1) * stride
#         available = max(available, 1)
#         if available <= 4:
#             start_index = 0
#         else:
#             start_index = np.random.randint(0, available - 3)
#         frame_indices = start_index + np.arange(self.sample_n_frames) * stride

#         # 读取视频 A
#         video = video_reader.get_batch(frame_indices).asnumpy()  # F, H, W, C
#         video = [Image.fromarray(frame) for frame in video]
#         pixel_values = [self.train_video_transforms(frame) for frame in video]
#         pixel_values = torch.stack(pixel_values)  # F, C, H, W

#         # 读取视频 B (GT)
#         video2 = video_reader2.get_batch(frame_indices).asnumpy()
#         video2 = [Image.fromarray(frame) for frame in video2]
#         pixel_values2 = [self.train_video_transforms(frame) for frame in video2]
#         pixel_values2 = torch.stack(pixel_values2)  # F, C, H, W

#         # 文本：优先读取 video_root2 同名 .txt；没有则返回空字符串
#         prompt = self._load_caption(video_path2)

#         sample = {
#             "pixel_values": pixel_values.permute(1, 0, 2, 3),       # C, F, H, W
#             "pixel_values2": pixel_values2.permute(1, 0, 2, 3),     # C, F, H, W
#             "prompts": prompt,
#         }
#         if self.use_first:
#             sample["first_frames"] = first_frame  # C, H, W

#         return sample
    

# #根据我的目录结构修改的版本
# class CustomDataset(Dataset):
#     def __init__(
#         self,
#         video_root,
#         video_root2,
#         first_root,
#         height=512,
#         width=512,
#         sample_n_frames=49,
#         is_one2three=False,
#         training_len=-1,
#         caption_ext=".txt",
#     ):
#         self.training_len = training_len
#         self.is_one2three = is_one2three

#         self.video_root = video_root   # Human (Remy-mixamo)
#         self.video_root2 = video_root2 # Robot (robot-videos/category)
#         self.first_root = first_root
#         self.caption_ext = caption_ext

#         print(f"[Dataset] Human Root: {self.video_root}")
#         print(f"[Dataset] Robot Root: {self.video_root2}")

#         self.video_exts = (".mp4", ".avi", ".mov", ".mkv")
#         self.img_exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp")

#         # =========================================================
#         # 核心修改：递归遍历 + 相对路径匹配
#         # =========================================================
#         self.video_paths, self.video_paths2, self.first_paths = self._walk_and_match()

#         self.use_first = (len(self.first_paths) > 0)
        
#         print(f"[Dataset] Found {len(self.video_paths)} paired videos.")
#         if self.use_first:
#             print(f"[Dataset] Found {len(self.first_paths)} corresponding first frames.")

#         self.height = height
#         self.width = width
#         self.train_video_transforms = transforms.Compose(
#             [
#                 transforms.Resize((height, width)), # 修改为 Resize 以适应不同分辨率视频
#                 # transforms.CenterCrop((height, width)),
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
#             ]
#         )

#         self.sample_n_frames = sample_n_frames
        
#         # 再次确认长度
#         self.len_videos = len(self.video_paths)
#         if self.len_videos == 0:
#              raise RuntimeError(f"No paired videos found between {video_root} and {video_root2}!")

#     def _walk_and_match(self):
#         """
#         递归遍历 video_root (Human)，获取相对路径 (如 scene01/angle01/vid.mp4)。
#         检查 video_root2 (Robot) 和 first_root 下是否存在对应的文件。
#         """
#         pairs_human = []
#         pairs_robot = []
#         pairs_first = []
        
#         # 检查首帧目录是否有效
#         check_first = (self.first_root is not None and os.path.isdir(self.first_root))

#         # 遍历 Human 目录
#         for root, _, files in os.walk(self.video_root):
#             for filename in files:
#                 if filename.lower().endswith(self.video_exts):
#                     # 1. 获取 Human 完整路径
#                     human_path = os.path.join(root, filename)
                    
#                     # 2. 获取相对路径 (关键一步)
#                     # 例如: absolute/path/to/Remy-mixamo/scene01/angle01/a.mp4
#                     # 变为: scene01/angle01/a.mp4
#                     rel_path = os.path.relpath(human_path, self.video_root)
                    
#                     # 3. 构造 Robot 完整路径
#                     robot_path = os.path.join(self.video_root2, rel_path)
                    
#                     # 4. 只有当 Robot 视频也存在时，才算配对成功
#                     if os.path.exists(robot_path):
#                         pairs_human.append(human_path)
#                         pairs_robot.append(robot_path)
                        
#                         # 5. 处理首帧 (假设首帧也在同样的子目录结构下，或者你需要根据实际情况调整)
#                         if check_first:
#                             # 尝试把视频后缀换成图片后缀找找看
#                             found_img = None
#                             base_rel_path = os.path.splitext(rel_path)[0] # 去掉 .mp4
                            
#                             # 在 first_root/scene01/angle01/ 下找同名图片
#                             for ext in self.img_exts:
#                                 img_path_guess = os.path.join(self.first_root, base_rel_path + ext)
#                                 if os.path.exists(img_path_guess):
#                                     found_img = img_path_guess
#                                     break
                            
#                             # 如果没找到同名图片，可能你的逻辑是直接取 robot 视频的首帧？
#                             # 这里暂定为：必须有图片文件才加入，否则填 None (后续处理要注意)
#                             # 为了保持对齐，如果开启了 use_first 但没找到图片，这组数据建议丢弃，或者报错
#                             if found_img:
#                                 pairs_first.append(found_img)
#                             else:
#                                 # 如果你的数据集不是必须有图片，这里可以 append(None) 并修改 __getitem__ 处理 None
#                                 # 但为了简单，这里假设如果开启首帧训练，就必须每一对都有首帧
#                                 # 如果找不到首帧，就把刚才加进去的视频也弹出来（丢弃这组）
#                                 pairs_human.pop()
#                                 pairs_robot.pop()

#         # 必须排序，保证多卡训练或不同 Seed 下顺序的一致性基础
#         # (shuffle 由 DataLoader 负责)
#         # 只要列表里的对应关系是正确的即可
        
#         # 打包排序，确保三个列表同步打乱/排序
#         if check_first and len(pairs_first) > 0:
#             combined = sorted(zip(pairs_human, pairs_robot, pairs_first))
#             pairs_human = [x[0] for x in combined]
#             pairs_robot = [x[1] for x in combined]
#             pairs_first = [x[2] for x in combined]
#         else:
#             combined = sorted(zip(pairs_human, pairs_robot))
#             pairs_human = [x[0] for x in combined]
#             pairs_robot = [x[1] for x in combined]
#             pairs_first = []

#         return pairs_human, pairs_robot, pairs_first

#     def __len__(self):
#         if self.training_len != -1:
#             return self.training_len
#         return len(self.video_paths)

#     def _caption_path_for(self, video2_path: str):
#         stem, _ = os.path.splitext(video2_path)
#         return stem + self.caption_ext

#     def _load_caption(self, video2_path: str) -> str:
#         cap_path = self._caption_path_for(video2_path)
#         if os.path.exists(cap_path) and os.path.isfile(cap_path):
#             try:
#                 with open(cap_path, "r", encoding="utf-8") as f:
#                     return f.read().strip()
#             except Exception:
#                 return ""
#         return ""

#     def __getitem__(self, index):
#         # 简单取模防止越界 (虽然 logic 上不会)
#         index = index % len(self.video_paths)

#         video_path = self.video_paths[index]
#         video_path2 = self.video_paths2[index] # 肯定是对应的

#         # 读取视频 A (Human)
#         video_reader = VideoReader(video_path)
#         video_length = len(video_reader)

#         # 读取视频 B (Robot GT)
#         video_reader2 = VideoReader(video_path2)
#         video_length2 = len(video_reader2)

#         # 读取首帧
#         first_frame = None
#         if self.use_first:
#             first_frame_path = self.first_paths[index]
#             first_frame = Image.open(first_frame_path).convert("RGB")
#             first_frame = self.train_video_transforms(first_frame)

#         # 只有在非常严格的情况下才 assert 长度完全相等
#         # 实际数据中，可能很难保证每一帧都对齐，通常取最小长度
#         real_len = min(video_length, video_length2)
        
#         assert self.sample_n_frames <= real_len, f"Video too short: {real_len} vs {self.sample_n_frames}. Path: {video_path}"

#         # 采样
#         stride = 1
#         available = real_len - (self.sample_n_frames - 1) * stride
#         available = max(available, 1)
#         if available <= 4:
#             start_index = 0
#         else:
#             start_index = np.random.randint(0, available - 3)
#         frame_indices = start_index + np.arange(self.sample_n_frames) * stride

#         # Get Batch A
#         video = video_reader.get_batch(frame_indices).asnumpy()
#         video = [Image.fromarray(frame) for frame in video]
#         pixel_values = [self.train_video_transforms(frame) for frame in video]
#         pixel_values = torch.stack(pixel_values)

#         # Get Batch B
#         video2 = video_reader2.get_batch(frame_indices).asnumpy()
#         video2 = [Image.fromarray(frame) for frame in video2]
#         pixel_values2 = [self.train_video_transforms(frame) for frame in video2]
#         pixel_values2 = torch.stack(pixel_values2)

#         prompt = self._load_caption(video_path2)

#         sample = {
#             "pixel_values": pixel_values.permute(1, 0, 2, 3),
#             "pixel_values2": pixel_values2.permute(1, 0, 2, 3),
#             "prompts": prompt,
#         }
#         if self.use_first:
#             sample["first_frames"] = first_frame

#         return sample
    

# -------------------------------------------------------------------
# 根据我的目录加上机器人外观参考图修改的版本
# -------------------------------------------------------------------
# class CustomDataset(Dataset):
#     def __init__(
#         self,
#         video_root,
#         video_root2,
#         robot_ref_path,  # <--- [修改] 变量名改一下，明确它是单张图
#         height=512,
#         width=512,
#         sample_n_frames=49,
#         is_one2three=False,
#         training_len=-1,
#         caption_ext=".txt",
#     ):
#         self.training_len = training_len
#         self.is_one2three = is_one2three

#         self.video_root = video_root
#         self.video_root2 = video_root2
#         self.robot_ref_path = robot_ref_path # 保存路径
#         self.caption_ext = caption_ext

#         self.video_exts = (".mp4", ".avi", ".mov", ".mkv")

#         # [逻辑变更] 1. 递归只负责匹配 Human 和 Robot 视频
#         self.video_paths, self.video_paths2 = self._walk_and_match()

#         # [逻辑变更] 2. 处理参考图
#         self.ref_frame_tensor = None
#         self.use_first = False
        
#         self.height = height
#         self.width = width
        
#         # 定义 transform
#         self.train_video_transforms = transforms.Compose([
#             transforms.Resize((height, width)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
#         ])

#         # [核心] 如果提供了参考图路径且文件存在，预加载它
#         if self.robot_ref_path and os.path.isfile(self.robot_ref_path):
#             print(f"[Dataset] Loading Static Reference Image: {self.robot_ref_path}")
#             try:
#                 # 加载图片
#                 ref_img = Image.open(self.robot_ref_path).convert("RGB")
#                 # 立即做变换并转为 Tensor (C, H, W)
#                 # 这样训练时不用每次 IO 读取，速度更快
#                 self.ref_frame_tensor = self.train_video_transforms(ref_img)
#                 self.use_first = True
#             except Exception as e:
#                 print(f"[WARN] Failed to load reference image: {e}")
#                 self.use_first = False
#         else:
#             print(f"[Dataset] No reference image found or path is empty.")

#         self.sample_n_frames = sample_n_frames
#         self.len_videos = len(self.video_paths)
#         if self.len_videos == 0:
#              raise RuntimeError(f"No paired videos found!")

#     def _walk_and_match(self):
#         """
#         仅匹配 Human 和 Robot 视频，不再查找 first_frames
#         """
#         pairs_human = []
#         pairs_robot = []
        
#         for root, _, files in os.walk(self.video_root):
#             for filename in files:
#                 if filename.lower().endswith(self.video_exts):
#                     human_path = os.path.join(root, filename)
#                     rel_path = os.path.relpath(human_path, self.video_root)
#                     robot_path = os.path.join(self.video_root2, rel_path)
                    
#                     if os.path.exists(robot_path):
#                         pairs_human.append(human_path)
#                         pairs_robot.append(robot_path)
#                         # [删除] 以前这里会去校验首帧是否存在，现在不需要了

#         # 排序保证一致性
#         combined = sorted(zip(pairs_human, pairs_robot))
#         pairs_human = [x[0] for x in combined]
#         pairs_robot = [x[1] for x in combined]
        
#         return pairs_human, pairs_robot
    
#     def __len__(self):
#         if self.training_len != -1:
#             return self.training_len
#         return len(self.video_paths)

#     def _caption_path_for(self, video2_path: str):
#         stem, _ = os.path.splitext(video2_path)
#         return stem + self.caption_ext

#     def _load_caption(self, video2_path: str) -> str:
#         cap_path = self._caption_path_for(video2_path)
#         if os.path.exists(cap_path) and os.path.isfile(cap_path):
#             try:
#                 with open(cap_path, "r", encoding="utf-8") as f:
#                     return f.read().strip()
#             except Exception:
#                 return ""
#         return ""
    
#     def __getitem__(self, index):
#         index = index % len(self.video_paths)
#         video_path = self.video_paths[index]
#         video_path2 = self.video_paths2[index]

#         # 读取视频 A (Human)
#         video_reader = VideoReader(video_path)
#         video_length = len(video_reader)
#         # 读取视频 B (Robot GT)
#         video_reader2 = VideoReader(video_path2)
#         video_length2 = len(video_reader2)

#         real_len = min(video_length, video_length2)
        
#         # 采样
#         stride = 1
#         available = real_len - (self.sample_n_frames - 1) * stride
#         available = max(available, 1)
#         if available <= 4:
#             start_index = 0
#         else:
#             start_index = np.random.randint(0, available - 3)
#         frame_indices = start_index + np.arange(self.sample_n_frames) * stride

#         # Get Batch A
#         video = video_reader.get_batch(frame_indices).asnumpy()
#         video = [Image.fromarray(frame) for frame in video]
#         pixel_values = [self.train_video_transforms(frame) for frame in video]
#         pixel_values = torch.stack(pixel_values)

#         # Get Batch B
#         video2 = video_reader2.get_batch(frame_indices).asnumpy()
#         video2 = [Image.fromarray(frame) for frame in video2]
#         pixel_values2 = [self.train_video_transforms(frame) for frame in video2]
#         pixel_values2 = torch.stack(pixel_values2)

#         prompt = self._load_caption(video_path2)

#         sample = {
#             "pixel_values": pixel_values.permute(1, 0, 2, 3),
#             "pixel_values2": pixel_values2.permute(1, 0, 2, 3),
#             "prompts": prompt,
#         }

#         # [修改] 直接使用预加载的 Tensor
#         if self.use_first:
#             # 这里的 ref_frame_tensor 已经是 [C, H, W] 且归一化过的
#             sample["first_frames"] = self.ref_frame_tensor

#         return sample


# # ... (上面是你修改后的 CustomDataset 类定义) ...

# if __name__ == "__main__":
#     import argparse
    
#     # 1. 简单的命令行参数，方便你动态传入路径测试
#     # 你可以直接在代码里写死路径，也可以用 python custom_dataset.py --human_root xxx 运行
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--human_root", type=str, 
#                         default="/opt/liblibai-models/user-workspace2/users/dxy/fyp/WorldWander/struc-data/Remy-mixamo",
#                         help="Path to Human videos (recursive)")
#     parser.add_argument("--robot_root", type=str, 
#                         default="/opt/liblibai-models/user-workspace2/users/dxy/fyp/WorldWander/struc-data/robot-videos/fourier-gr2",
#                         help="Path to Robot videos (recursive)")
#     parser.add_argument("--first_root", type=str, 
#                         default= "/opt/liblibai-models/user-workspace2/users/dxy/robotic/InteractionVideo/robot_ref_img/unitree-g1.jpg", 
#                         help="Path to first frames (optional)")
    
#     args = parser.parse_args()

#     print("="*50)
#     print("Dataset Sanity Check")
#     print("="*50)

#     # 2. 实例化 Dataset
#     try:
#         dataset = CustomDataset(
#             video_root=args.human_root,
#             video_root2=args.robot_root,
#             robot_ref_path=args.first_root,
#             height=224, width=416,  # 这里用小一点的分辨率测试速度
#             sample_n_frames=16,     # 采样 16 帧
#             is_one2three=True
#         )
#     except Exception as e:
#         print(f"\n[FATAL ERROR] Dataset init failed: {e}")
#         exit(1)

#     print(f"\n[Stats] Total Samples: {len(dataset)}")
    
#     # 3. 检查内部配对列表 (前5个)
#     print("\n[Check Pairing Logic] First 5 pairs:")
#     limit = min(5, len(dataset.video_paths))
#     for i in range(limit):
#         h_path = dataset.video_paths[i]
#         r_path = dataset.video_paths2[i]
        
#         # 提取相对路径方便对比
#         rel_h = os.path.relpath(h_path, args.human_root)
#         rel_r = os.path.relpath(r_path, args.robot_root)
        
#         match_status = "✅ MATCH" if rel_h == rel_r else "❌ MISMATCH"
#         print(f"  [{i}] {match_status}")
#         print(f"      Human: .../{rel_h}")
#         print(f"      Robot: .../{rel_r}")
#         print(f"      Human: {h_path}")
#         print(f"      Robot: {r_path}")

#     # 4. 尝试加载数据 (测试 __getitem__)
#     print("\n[Test Loading] Trying to load first 3 batches...")
    
#     # 创建一个简单的 DataLoader
#     from torch.utils.data import DataLoader
#     loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0) # num_workers=0 方便调试报错
    
#     try:
#         for i, batch in enumerate(loader):
#             if i >= 3: break
            
#             print(f"\n  Batch {i}:")
#             pixel_values = batch["pixel_values"]
#             pixel_values2 = batch["pixel_values2"]
#             prompts = batch["prompts"]
            
#             print(f"    - Human Shape: {pixel_values.shape} (B, C, F, H, W)")
#             print(f"    - Robot Shape: {pixel_values2.shape} (B, C, F, H, W)")
#             print(f"    - Prompts: {prompts}")
            
#             if "first_frames" in batch:
#                 print(f"    - First Frame: {batch['first_frames'].shape} (B, C, H, W)")
            
#             # 简单的数值范围检查
#             print(f"    - Range Human: [{pixel_values.min():.2f}, {pixel_values.max():.2f}] (Should be approx -1 to 1)")
            
#     except Exception as e:
#         print(f"\n[FATAL ERROR] Data loading failed at batch {i}: {e}")
#         import traceback
#         traceback.print_exc()
#         exit(1)

#     print("\n" + "="*50)
#     print("✅ TEST PASSED: Dataset seems ready for training.")
#     print("="*50)



# 使用 JSONL 文件作为数据源的版本
# custom_dataset.py
import json
import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from decord import VideoReader
from torchvision import transforms
from PIL import Image

class CustomDataset(Dataset):
    def __init__(
        self,
        jsonl_path,
        width=512,
        height=512,
        sample_n_frames=49,
        is_one2three=False, 
        training_len=-1,
        # === [修改点] 键名配置 ===
        # 根据您的要求：control_video 是条件，video 是目标
        cond_key="control_video",   # 对应 pixel_values (Condition)
        target_key="video",         # 对应 pixel_values2 (Target)
        prompt_key="prompt",
        ref_key="ref_image_path"    # 目标外观参考图
    ):
        self.jsonl_path = jsonl_path
        self.training_len = training_len
        self.cond_key = cond_key
        self.target_key = target_key
        self.prompt_key = prompt_key
        self.ref_key = ref_key
        
        print(f"[Dataset] Loading metadata from {jsonl_path} ...")
        self.data_list = []
        try:
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        self.data_list.append(json.loads(line))
        except Exception as e:
            raise RuntimeError(f"Failed to read JSONL file: {e}")

        print(f"[Dataset] Found {len(self.data_list)} samples.")
        print(f"[Dataset] Mapping: Condition='{self.cond_key}' -> Target='{self.target_key}'")
        
        self.height = height
        self.width = width
        self.sample_n_frames = sample_n_frames
        
        self.train_video_transforms = transforms.Compose([
            transforms.Resize((height, width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        if self.training_len != -1:
            return self.training_len
        return len(self.data_list)

    def __getitem__(self, index):
        # 循环取模
        item = self.data_list[index % len(self.data_list)]
        
        # === 1. 解析路径 ===
        cond_path = item.get(self.cond_key)     # 您的 'control_video'
        target_path = item.get(self.target_key) # 您的 'video'
        prompt = item.get(self.prompt_key, "")
        ref_img_path = item.get(self.ref_key)

        # 校验文件是否存在
        if not cond_path or not os.path.exists(cond_path):
            # 简单的容错策略：如果找不到文件，就随机换一行数据
            # print(f"[Warn] Missing condition video: {cond_path}, skipping...")
            return self.__getitem__((index + 1) % len(self.data_list))
            
        if not target_path or not os.path.exists(target_path):
            # print(f"[Warn] Missing target video: {target_path}, skipping...")
            return self.__getitem__((index + 1) % len(self.data_list))

        # === 2. 加载视频 ===
        try:
            vr_cond = VideoReader(cond_path)
            vr_target = VideoReader(target_path)
        except Exception as e:
            print(f"[Error] Failed loading video: {cond_path} or {target_path}")
            return self.__getitem__((index + 1) % len(self.data_list))
            
        len_cond = len(vr_cond)
        len_target = len(vr_target)
        real_len = min(len_cond, len_target)

        # === 3. 帧采样 ===
        if real_len < self.sample_n_frames:
             # 如果视频太短，简单跳过（实际生产中可能需要 padding 策略）
             return self.__getitem__((index + 1) % len(self.data_list))

        stride = 1
        available = real_len - (self.sample_n_frames - 1) * stride
        available = max(available, 1)
        
        if available <= 4:
            start_index = 0
        else:
            start_index = np.random.randint(0, available - 3)
            
        frame_indices = start_index + np.arange(self.sample_n_frames) * stride
        frame_indices = np.clip(frame_indices, 0, real_len - 1)

        # === 4. 获取 Tensor ===
        # Condition (control_video) -> pixel_values
        frames_cond = vr_cond.get_batch(frame_indices).asnumpy()
        frames_cond = [Image.fromarray(f) for f in frames_cond]
        pixel_values = torch.stack([self.train_video_transforms(f) for f in frames_cond]) # [F, C, H, W]

        # Target (video) -> pixel_values2
        frames_target = vr_target.get_batch(frame_indices).asnumpy()
        frames_target = [Image.fromarray(f) for f in frames_target]
        pixel_values2 = torch.stack([self.train_video_transforms(f) for f in frames_target]) # [F, C, H, W]

        # === 5. 处理首帧/参考图 (Target的外观参考) ===
        first_frame_tensor = torch.zeros((3, self.height, self.width)) 
        
        if ref_img_path and os.path.exists(ref_img_path):
            try:
                ref_img = Image.open(ref_img_path).convert("RGB")
                first_frame_tensor = self.train_video_transforms(ref_img) # [C, H, W]
            except Exception as e:
                print(f"[Warn] Bad ref image: {ref_img_path}")
        
        # === 6. 返回 ===
        sample = {
            "pixel_values": pixel_values.permute(1, 0, 2, 3),   # [C, F, H, W] (Condition)
            "pixel_values2": pixel_values2.permute(1, 0, 2, 3), # [C, F, H, W] (Target)
            "prompts": prompt,
            "first_frames": first_frame_tensor, # [C, H, W]

            # [新增] 将路径放入返回字典，用于调试和追溯
            "cond_path": cond_path,
            "target_path": target_path,
            "prompt_path": prompt,
            "ref_path": ref_img_path if ref_img_path else "None"
        }

        return sample


if __name__ == "__main__":
    import argparse
    import torchvision
    from torch.utils.data import DataLoader

    # 1. 设置参数
    parser = argparse.ArgumentParser(description="Test CustomDataset with JSONL")
    parser.add_argument("--jsonl", type=str, default="./data/second_phrase/json_outputs/data_unitree-g1.jsonl", 
                        help="Path to the jsonl file to test")
    parser.add_argument("--batch_size", type=int, default=2)
    args = parser.parse_args()

    print("="*50)
    print(f"Starting Dataset Sanity Check...")
    print(f"Target JSONL: {args.jsonl}")
    print("="*50)

    # 2. 实例化 Dataset
    try:
        dataset = CustomDataset(
            jsonl_path=args.jsonl,
            width=416,          
            height=224,         
            sample_n_frames=16, 
            cond_key="control_video",  
            target_key="video",        
        )
    except Exception as e:
        print(f"\n[FATAL] Dataset init failed: {e}")
        exit(1)

    print(f"\n[Stats] Dataset Length: {len(dataset)}")

    # 3. 实例化 DataLoader
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # 4. 获取一个 Batch 并打印信息
    try:
        for i, batch in enumerate(loader):
            print(f"\n[Batch {i} Info]")
            
            cond = batch["pixel_values"]
            target = batch["pixel_values2"]
            prompts = batch["prompts"]
            refs = batch["first_frames"]
            
            # [新增] 提取路径列表
            cond_paths = batch.get("cond_path", [])
            target_paths = batch.get("target_path", [])
            ref_paths = batch.get("ref_path", [])

            print(f"  > Condition (Human) Shape : {cond.shape}")  
            print(f"  > Target (Robot) Shape    : {target.shape}")
            print(f"  > Ref Image Shape         : {refs.shape}")
            print(f"  > Value Range (Cond)      : [{cond.min():.2f}, {cond.max():.2f}]")
            
            # [新增] 打印这个 Batch 里面具体包含了哪些文件
            print("\n  [Loaded File Paths]")
            for b_idx in range(len(cond_paths)):
                print(f"    Sample {b_idx}:")
                print(f"      - Human (Cond) : {cond_paths[b_idx]}")
                print(f"      - Robot (Tgt)  : {target_paths[b_idx]}")
                print(f"      - Ref Image    : {ref_paths[b_idx]}")
                print(f"      - Prompt       : {prompts[b_idx][:50]}...") # 截断打印提示词

            # --- 可视化保存 ---
            def denorm(x):
                return (x * 0.5 + 0.5).clamp(0, 1)

            img_cond = denorm(cond[0, :, 0, :, :]) 
            img_target = denorm(target[0, :, 0, :, :])
            img_ref = denorm(refs[0, :, :, :])

            combined = torch.cat([img_cond, img_target, img_ref], dim=2) 
            
            save_path = "debug_vis_check.jpg"
            torchvision.utils.save_image(combined, save_path)
            print(f"\n[Visual Check] Saved a preview image of Sample 0 to: {save_path}")
            print("  (Left: Human/Cond, Middle: Robot/Target, Right: Ref Image)")

            break # 只测一个 batch 就退出

    except Exception as e:
        print(f"\n[FATAL] DataLoader failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    print("\n" + "="*50)
    print("✅ TEST PASSED: Dataset works correctly.")
    print("="*50)