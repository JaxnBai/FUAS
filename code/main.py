import  shutil, os
import sys, getopt
import imageio,json,requests
from tqdm import tqdm
import nibabel as nib
import numpy as np
import  argparse,json
import random
import shutil
from PIL import Image
from dose_predict import DosePredictor
# import fit2
# sys.path.insert(0, '/root/autodl-tmp/Medical-SAM2/')
from segmentation import Segconfig



class ArgsConfig:
    def __init__(self,args):
        self.args =args
  
    def niito2D(self,full_input_path, outputpath,id):

        output_image=outputpath #os.path.join(outputpath,"image/")   
        if not os.path.exists(output_image):
            os.mkdir(output_image)

        if not self.is_gzip_file(full_input_path):
            print(f"Warning: {full_input_path} is not a valid gzip file.")
            # continue  
        # breakpoint()
        image = nib.load(full_input_path)
        image_array = image.get_fdata()  
        # print(len(image_array.shape))
        (x, y ,z) = image_array.shape  # 
        #print("(x,y,z):",x,y,z)
        loop = tqdm(range(0, z), desc="Converting ") # 
        
        image_dir=os.path.join(output_image,id)
        if not os.path.exists(image_dir):
            os.mkdir(image_dir)

        for current_slice in loop:
            loop.set_postfix({ "Path": os.path.basename(full_input_path)}, refresh=True)
            temp_image_path = "{}".format(str(current_slice )) + ".png"
            if os.path.exists(os.path.join(image_dir,temp_image_path)):
                continue
            data = image_array[:,:, current_slice]  
            data = (data - np.min(data)) / (np.max(data) - np.min(data)) * 255  # 归一化
            data = data.astype(np.uint8)
            # data = cv2.equalizeHist(data)  

            #print(data.shape)
        
            
            imageio.imwrite(temp_image_path, data)

            shutil.move(temp_image_path, image_dir)
                
                

    def nii_to_image(self):
        niigz_path=self.args.niigz_path
        ids=os.listdir(niigz_path)
        loop=tqdm(ids,desc='Converting All')
        for id in loop:
            loop.set_postfix({ "ID": id}, refresh=True)
            sub_root = os.path.join(niigz_path, id)
            for root, dirs, files in os.walk(sub_root):
                for file in files:
                    if file.endswith('.nii.gz'):
                        
                        full_path = os.path.join(root, file)                      
                        
                        try:
                            parts = full_path.split('/')
                            modality = parts[-5]              # e.g., AP
                            uid = parts[-4]                   # e.g., 330E96BB-...
                            t1_folder = parts[-2]             # e.g., t1
                            original_filename = file          # e.g., 701_t1_quick3d_...
                            serial = original_filename.split('_')[0]  # 701
                            # 构造目标文件名
                            new_filename = f"{id[:6]}_{modality}_{t1_folder}_{serial}"
                            if modality=='BP' and t1_folder=='t2':
                                self.niito2D(full_path, self.args.image_path, new_filename)

                                os.makedirs(self.args.mri_input, exist_ok=True)
                                if not os.path.exists(os.path.join(self.args.mri_input,new_filename)):
                                    os.makedirs(os.path.join(self.args.mri_input,new_filename))
                                    shutil.copy(os.path.join(niigz_path,id,'mri.txt'),os.path.join(self.args.mri_input,new_filename,'mri.txt'))

                        except Exception as e:
                            raise RuntimeError(f" {full_path}，错误信息: {e}")      

    def image_to_nii(self):
        args = self.args
        root_dir = args.image_path
        output_dir = args.image_nii_path
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # 遍历每个子文件夹
        for folder_name in tqdm(sorted(os.listdir(root_dir)),desc="Converting image to nii file"):
            folder_path = os.path.join(root_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue

            # 读取并排序所有png文件
            png_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.png')],
                            key=lambda x: int(x.replace('.png', '')))
            
            slices = []
            for png in png_files:
                img = Image.open(os.path.join(folder_path, png)).convert('L')  # 转灰度图
                img = img.resize((1024, 1024))  # 可选，若图像大小不一致
                slices.append(np.array(img))
            
            volume = np.stack(slices, axis=-1)  # 形成3D体数据 (H, W, D)
            affine = np.eye(4)  # 默认空间坐标变换矩阵

            nii_img = nib.Nifti1Image(volume, affine)
            nib.save(nii_img, os.path.join(output_dir, f'{folder_name}.nii.gz'))

    def mask_to_nii(self):
        args = self.args
        input_root = args.pred_mask
        output_dir = args.mask_nii_path
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)    

        # 遍历每个子文件夹
        for folder_name in tqdm(sorted(os.listdir(input_root)),desc="Converting mask to nii file"):
            folder_path = os.path.join(input_root, folder_name)
            if not os.path.isdir(folder_path):
                continue

            # 获取所有 .npy 文件并按数字顺序排序
            npy_files = sorted(
                [f for f in os.listdir(folder_path) if f.endswith('.npy')],
                key=lambda x: int(x.replace('.npy', ''))
            )

            slices = []
            for file in npy_files:
                slice_array = np.load(os.path.join(folder_path, file))
                slices.append(slice_array)

            # 将多个切片堆叠成一个3D体
            volume = np.stack(slices, axis=-1)  # Shape: (H, W, D)
            affine = np.eye(4)  # 默认仿射矩阵

            # 保存为 nii.gz
            nii_img = nib.Nifti1Image(volume, affine)
            nib.save(nii_img, os.path.join(output_dir, f'{folder_name}.nii.gz'))

    def is_gzip_file(self,filepath):
        try:
            with open(filepath, 'rb') as f:
                return f.read(2) == b'\x1f\x8b'
        except Exception as e:
            print(f"Error checking file: {e}")
            return False

    def resize_images(self,output_size=(1024, 1024) ):
        input_folders=self.args.image_path
        save_to=self.args.image_newsize_path
        ids=os.listdir(input_folders)
        loop=tqdm(ids,desc='Resizing')
        for id in loop:
            input_folder = os.path.join(input_folders,id)
            if not os.path.isdir(input_folder):
                print(f"文件夹 {input_folder} 不存在，跳过。")
                continue
            pred_mask=os.path.join(save_to,id)
            if not os.path.exists(pred_mask):
                os.makedirs(pred_mask, exist_ok=True)

            for file in sorted(os.listdir(input_folder)):
                if file.endswith('.png'):
                    img_path = os.path.join(input_folder, file)
                    img = Image.open(img_path)

                    # Resize 并保存
                    img_resized = img.resize(output_size, Image.BILINEAR)
                    img_resized.save(os.path.join(pred_mask, file))
        print(f"All images have been resized to {output_size} and saved to {save_to}")

    def save(self,collected_info):
        args= self.args
        final_path= args.final_path 
        if not os.path.exists(final_path):
            os.mkdir(final_path)
        
        for patient_id,slice_idx,predicted_dose in collected_info:
            dir_path=os.path.join(final_path,f'{patient_id}')
            if not os.path.exists(dir_path):
                os.mkdir(dir_path)
            shutil.copy(os.path.join(args.visual_path,patient_id,f'{slice_idx}.png'),os.path.join(dir_path,f'{slice_idx}_predict.png'))
            shutil.copy(os.path.join(args.image_path,patient_id,f'{slice_idx}.png'),os.path.join(dir_path,f'{slice_idx}_origin.png'))
            shutil.copy(os.path.join(args.image_newsize_path,patient_id,f'{slice_idx}.png'),os.path.join(dir_path,f'{slice_idx}_1024*1024.png'))
            shutil.copy(os.path.join(args.pred_mask,patient_id,f'{slice_idx}.npy'),os.path.join(dir_path,f'{slice_idx}_mask.npy'))
            shutil.copy(os.path.join(args.mri_output,patient_id,'mri_raw.txt'),os.path.join(dir_path,f'mri_raw.txt'))
            shutil.copy(os.path.join(args.mri_output,patient_id,'mri_optimized.txt'),os.path.join(dir_path,f'mri_optimized.txt'))
            with open(os.path.join(dir_path,f'{slice_idx}_dose_prediction.txt'), 'w') as f:
                f.write(f"Predicted Dose: {predicted_dose:.2f} J\n")
                f.write(f"Patient ID: {patient_id}\n")
                f.write(f"Slice Index: {slice_idx}\n")
        print(f"all results have been saved to {final_path}")
   
class Visualize:
    def __init__(self,npy_folder, png_folder, output_folder, vis_name,alpha=0.6):
        self.png_folder = png_folder
        self.npy_folder = npy_folder
        self.output_folder = output_folder
        self.vis_name = vis_name
        self.alpha = alpha
        self.label_colors = {
                            0: (0, 0, 0),
                            1: (255, 0, 0),
                            2: (0, 255, 0),
                            3: (0, 0, 255),
                            4: (255, 255, 0),
                            5: (0, 255, 255),
                            6: (255, 0, 255),
                            7: (255, 255, 255)
                        }

    def create_colorful_mask(self,npy_data):
        colored = np.zeros((npy_data.shape[0], npy_data.shape[1], 3), dtype=np.uint8)
        for label_id, color in self.label_colors.items():
            mask = npy_data == label_id
            colored[mask] = color
        return Image.fromarray(colored)

    def overlay_npy_on_png(self,npy_path, png_path, pred_mask, alpha):

        png_img = Image.open(png_path).convert("RGBA")
        npy_data = np.load(npy_path)
        npy_img = Image.fromarray(npy_data)

        # 调整 npy 图像的尺寸以匹配 png 图像
        if png_img.size != npy_img.size:
            npy_img = npy_img.resize(png_img.size)

        # 将灰度数据转换为彩色掩码
        color_mask = self.create_colorful_mask(np.array(npy_img))
        color_mask = color_mask.convert("RGBA")
        # 设置掩码的透明度
        r, g, b, a = color_mask.split()
        a = a.point(lambda p: p * alpha)
        color_mask = Image.merge('RGBA', (r, g, b, a))

        # 将彩色掩码叠加到 png 图像上
        combined = Image.alpha_composite(png_img, color_mask)
        combined.save(pred_mask)
        # print(f"已保存结果到 {pred_mask}")

    def process_folders(self):
        npy_folder = self.npy_folder
        png_folder = self.png_folder
        output_folder = self.output_folder
        alpha = self.alpha

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        if not os.path.exists(npy_folder):
            print(f"错误:snpy 文件夹 {npy_folder} 不存在。")
            return

        # 获取 mask 文件夹下的所有小文件夹
        sub_npy_folders = [f for f in os.listdir(png_folder) if os.path.isdir(os.path.join(png_folder, f))]
        # breakpoint()
        for sub_npy_folder in tqdm(sub_npy_folders, desc="Visualizing"):
            
            sub_npy_folder_path = os.path.join(npy_folder, sub_npy_folder)
            sub_png_folder_path = os.path.join(png_folder, sub_npy_folder)
            if not os.path.exists(sub_png_folder_path):
                print(f"错误：对应的 png 文件夹 {sub_png_folder_path} 不存在。")
                continue
            sub_output_folder = os.path.join(output_folder, sub_npy_folder) ################################################################################

            # 为每个小文件夹创建对应的输出子文件夹
            if not os.path.exists(sub_output_folder):
                os.makedirs(sub_output_folder)

            npy_files = [f for f in os.listdir(sub_npy_folder_path) if f.endswith('.npy')]
            # 内层进度条，显示小文件夹内文件的处理进度
            for npy_filename in tqdm(npy_files, desc=f"Processing {sub_npy_folder} ", leave=False):
                npy_path = os.path.join(sub_npy_folder_path, npy_filename)
                base_name = os.path.splitext(npy_filename)[0]
                png_filename = f"{base_name}.png"
                png_path = os.path.join(sub_png_folder_path, png_filename)
                if os.path.exists(png_path):
                    output_filename = f"{base_name}.png"
                    pred_mask = os.path.join(sub_output_folder, output_filename)
                    self.overlay_npy_on_png(npy_path, png_path, pred_mask, alpha)
                else:
                    breakpoint()
                    print(f"未找到对应的 .png 文件: {png_filename}")
        principles = """
        治疗计划制定的基本原则
        一、病史相关
        1.有3个月内人流史，不建议行HIFUA手术；
        2.有节育环，需在HIFUA术前3天取出；
        3.有盆腔炎，不建议行HIFUA手术；
        4.有听力/交流障碍，不建议行HIFUA手术；
        5.有下腹部手术史，需判断是否有严重肠粘连和瘢痕。当瘢痕宽度达到15mm及以上，不建议行HIFUA手术；
        二、治疗范围
        1.治疗区的边界与肌瘤的上下（头足）、左右边界之间的距离为5-10mm，与内膜间的距离≥15cm，与肌瘤深面边界和浅面边界（骶骨侧边界和腹壁侧边界）的距离为10mm；
        2.超声消融首先治疗的层面推荐为病灶左右径的1/2，前后径的1/2、上下径靠脚侧的1/4区域。其次是对超声治疗敏感的区域（容易出现灰度变化的区域，如钙化区、坏死区、缺血区等）；
        3.在团块状灰度出现前，依次逐层进行治疗。出现团块且灰度变化到达临近层面，可以间隔一个层面进行治疗；
        4.首先根据病人的反应调节剂量与强度，其次根据治疗靶区灰度变化进行调节。当出现团块状灰度变化，则根据团块状灰度扩散进行照射。没有出现团块状变化则按照剂量计划照射；
        三、肌瘤类型
        1.最大径小于20mm的肌瘤，使用掏心辐照；
        2.粘膜下肌瘤治疗时强调焦点到内膜的距离大于15mm；
        3.浆膜下肌瘤可先治疗肌瘤中心，再向周围扩散；
        4.T2WI高信号且血供丰富的肌瘤、最大径超过10cm的大肌瘤，建议多次治疗；
        四、治疗难度
        1.T2WI信号为等高信号的，治疗难度大，低信号治疗难度较小；
        2.肿瘤血供丰富的，治疗难度大，血供不丰富的难度较小；
        3.后位子宫，夹杂肠道，治疗难度大，前位子宫治疗难度相对小。
        五、其他
        1.什么时候使用大中小膀胱？
        一般前位子宫、前壁肌瘤、肌瘤较大的情况下使用较小膀胱，后位子宫情况下使用较大膀胱，后壁肌瘤大小膀胱都有在使用，小肌瘤的话还需结合子宫及肌瘤位置综合考虑。
        2.什么时候使用大中小水囊？
        水囊使用的目的主要是制造安全合适的声通道，如需要推挤肠道，一般水囊张力较高水囊大，如果是对声通道进行调整，如调整焦距和膀胱形态等，水囊可大可小。
        """        
  
class MriStratery:
    def __init__(self, args):
        self.args = args

    def call_model(self, prompt):
        url = "https://api.siliconflow.cn/v1/chat/completions"
        payload = {
            "model": "Qwen/Qwen3-14B",
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "max_tokens": 1024,
            "enable_thinking": True,
            "thinking_budget": 4096,
            "min_p": 0.05,
            "stop": None,
            "temperature": 0.7,
            "top_p": 0.7,
            "top_k": 50,
            "frequency_penalty": 0.5,
            "n": 1,
            "response_format": {"type": "text"},
        }
        with open('config.json', 'r') as file:
            config = json.load(file)

        api_key = config['qwen3_key']        
        headers = {
            "Authorization": api_key,  # 请替换成你的真实 key
            "Content-Type": "application/json"
        }

        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def inference(self):
        os.makedirs(self.args.mri_output, exist_ok=True)

        principles = """
        治疗计划制定的基本原则
        一、病史相关
        1.有3个月内人流史，不建议行HIFUA手术；
        2.有节育环，需在HIFUA术前3天取出；
        3.有盆腔炎，不建议行HIFUA手术；
        4.有听力/交流障碍，不建议行HIFUA手术；
        5.有下腹部手术史，需判断是否有严重肠粘连和瘢痕。当瘢痕宽度达到15mm及以上，不建议行HIFUA手术；
        二、治疗范围
        1.治疗区的边界与肌瘤的上下（头足）、左右边界之间的距离为5-10mm，与内膜间的距离≥15cm，与肌瘤深面边界和浅面边界（骶骨侧边界和腹壁侧边界）的距离为10mm；
        2.超声消融首先治疗的层面推荐为病灶左右径的1/2，前后径的1/2、上下径靠脚侧的1/4区域。其次是对超声治疗敏感的区域（容易出现灰度变化的区域，如钙化区、坏死区、缺血区等）；
        3.在团块状灰度出现前，依次逐层进行治疗。出现团块且灰度变化到达临近层面，可以间隔一个层面进行治疗；
        4.首先根据病人的反应调节剂量与强度，其次根据治疗靶区灰度变化进行调节。当出现团块状灰度变化，则根据团块状灰度扩散进行照射。没有出现团块状变化则按照剂量计划照射；
        三、肌瘤类型
        1.最大径小于20mm的肌瘤，使用掏心辐照；
        2.粘膜下肌瘤治疗时强调焦点到内膜的距离大于15mm；
        3.浆膜下肌瘤可先治疗肌瘤中心，再向周围扩散；
        4.T2WI高信号且血供丰富的肌瘤、最大径超过10cm的大肌瘤，建议多次治疗；
        四、治疗难度
        1.T2WI信号为等高信号的，治疗难度大，低信号治疗难度较小；
        2.肿瘤血供丰富的，治疗难度大，血供不丰富的难度较小；
        3.后位子宫，夹杂肠道，治疗难度大，前位子宫治疗难度相对小。
        五、其他
        1.什么时候使用大中小膀胱？
        一般前位子宫、前壁肌瘤、肌瘤较大的情况下使用较小膀胱，后位子宫情况下使用较大膀胱，后壁肌瘤大小膀胱都有在使用，小肌瘤的话还需结合子宫及肌瘤位置综合考虑。
        2.什么时候使用大中小水囊？
        水囊使用的目的主要是制造安全合适的声通道，如需要推挤肠道，一般水囊张力较高水囊大，如果是对声通道进行调整，如调整焦距和膀胱形态等，水囊可大可小。
        """        
 
        for patient_id in tqdm(os.listdir(self.args.mri_input), desc="Generating treatment plans"):
            patient_folder = os.path.join(self.args.mri_input, patient_id)
            mri_file_path = os.path.join(patient_folder, 'mri.txt')

            try:
                with open(mri_file_path, 'r') as f:
                    input_content = f.read()

                # === 第一次推理 ===
                first_prompt = (
                    principles + "\n\n"
                    + input_content
                    + "\n请你根据以上所有原则的内容帮我写治疗方案，治疗方案内容应包含：\n"
                    + "1.初步诊断：\n2.治疗时机：\n3.治疗目的及医患沟通：\n4.术前准备及注意事项：\n"
                    + "5.治疗分析及术中注意事项：\n6.治疗后观察和处理：\n开启模型思考模式。"
                )
                first_result = self.call_model(first_prompt)

                # === 第二次推理（优化） ===
                second_prompt = (
                    principles + "\n\n"
                    + input_content
                    + "\n以下是第一次生成的治疗方案：\n"
                    + first_result
                    + "\n请你在上述方案的基础上，进一步优化内容，使之更具临床可行性、语言更清晰严谨，必要时可做增补，结构保持一致。"
                )
                optimized_result = self.call_model(second_prompt)

                # === 保存结果 ===
                output_dir = os.path.join(self.args.mri_output, patient_id)
                os.makedirs(output_dir, exist_ok=True)

                with open(os.path.join(output_dir, 'mri_raw.txt'), 'w') as f:
                    f.write(first_result)

                with open(os.path.join(output_dir, 'mri_optimized.txt'), 'w') as f:
                    f.write(optimized_result)

            except Exception as e:
                raise RuntimeError(f"{mri_file_path} 处理失败，错误信息: {e}")                

class Planner:
    def __init__(self):

        self.args = self.parse_args()        
        output_dir=os.path.dirname(self.args.image_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        self.args.image_newsize_path = os.path.join(output_dir, 'image_newsize')
        self.args.visual_path = os.path.join(output_dir, 'visual_image')
        self.args.mask_nii_path = os.path.join(output_dir, 'mask_nii')
        self.args.image_nii_path = os.path.join(output_dir, 'image_nii')
        self.args.pred_mask = os.path.join(output_dir, 'pred_mask')
        self.args.final_path = os.path.join(output_dir, 'final')
        self.args.mri_input = os.path.join(output_dir, 'mri_input')
        self.args.mri_output = os.path.join(output_dir, 'mri_output')

        self.config = ArgsConfig(self.args)
        self.seg=Segconfig(self.args)
        self.visual=Visualize(self.args.pred_mask,self.args.image_newsize_path, self.args.visual_path, None,alpha=0.6)
        self.dose=DosePredictor(self.args)
        self.mri=MriStratery(self.args)

    def parse_args(self): 
        print("[DEBUG] Before adding arguments")

        parser = argparse.ArgumentParser()
        parser.add_argument('-net', type=str, default='sam2', help='net type')
        parser.add_argument('-encoder', type=str, default='vit_b', help='encoder type')
        parser.add_argument('-exp_name', default='BTCV_MedSAM2', type=str, help='experiment name')
        parser.add_argument('-vis', type=bool, default=False, help='Generate visualisation during validation')
        parser.add_argument('-train_vis', type=bool, default=False, help='Generate visualisation during training')
        parser.add_argument('-prompt', type=str, default='click', help='type of prompt, bbox or click')
        parser.add_argument('-prompt_freq', type=int, default=2, help='frequency of giving prompt in 3D images')
        parser.add_argument('-seg_model', type=str, default=None, help='path of seg_model weights')
        parser.add_argument('-val_freq',type=int,default=1,help='interval between each validation')
        parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
        parser.add_argument('-gpu_device', type=int, default=0, help='use which gpu')
        parser.add_argument('-image_size', type=int, default=1024, help='image_size')
        parser.add_argument('-out_size', type=int, default=1024, help='output_size')
        parser.add_argument('-distributed', default='none' ,type=str,help='multi GPU ids to use')
        parser.add_argument('-dataset', default='btcv' ,type=str,help='dataset name')
        parser.add_argument('-sam_ckpt', type=str, default='./checkpoints/sam2_hiera_small.pt' , help='sam checkpoint address')
        parser.add_argument('-sam_config', type=str, default='sam2_hiera_s' , help='sam checkpoint address')
        parser.add_argument('-video_length', type=int, default=2, help='sam checkpoint address')
        
        parser.add_argument('-weights', type=str, default = 0, help='the weights file you want to test')
        parser.add_argument('-multimask_output', type=int, default=1 , help='the number of masks output for multi-class segmentation')
        parser.add_argument('-memory_bank_size', type=int, default=16, help='sam 2d memory bank size')

        parser.add_argument('-requires_mask', type=str, default='True', help='is mask required when training') 
        parser.add_argument('-batch', type=int, default=1, help='batch size for dataloader')
        parser.add_argument('-firstlr', type=float, default=1e-4, help='initial sam_layers learning rate')
        parser.add_argument('-secondlr', type=float, default=1e-8, help='initial mem_layers learning rate')
        parser.add_argument('-exp', type=str, default='exp0.0', help='initial mem_layers learning rate')
        parser.add_argument('-seg_type', type=str, default='autonomy input', help='initial mem_layers learning rate')

        parser.add_argument(
        '-niigz_path',
        type=str,
        default='./20例影像科报告文本',   
        help='The path of segmentation output data')
    
        parser.add_argument(
        '-image_nii_path',
        type=str,
        default='./data/image_nii',
        help='The path of segmentation output data')
        
        parser.add_argument(
        '-mask_nii_path',
        type=str,
        default='./data/mask_nii',
        help='The path of segmentation output data')

        parser.add_argument(
        '-image_path',
        type=str,
        default='./data/image',
        help='The path of segmentation image')

        parser.add_argument(
        '-origin_mask',
        type=str,
        default='./data/test_mask',
        help='The path of segmentation origin_mask')

        parser.add_argument(
        '-image_newsize_path',
        type=str,
        default='./data/btcv',
        help='The path of segmentation data')

        parser.add_argument(
        '-visual_path',
        type=str,
        default='./data/btcv',
        help='The path of segmentation data')

        parser.add_argument(
        '-pred_mask',
        type=str,
        default='./pred_mask',
        help='The path of segmentation output data')

        parser.add_argument(
        '-final_path',
        type=str,
        default='./final_path',
        help='The path of segmentation output data')

        parser.add_argument(
        '-dose_csv_path',
        type=str,
        default='./total_id_doses1.csv',
        help='The path of segmentation output data')    

        parser.add_argument(
        '-dose_model',
        type=str,
        default='./dose_model/dose_model_BayesSearchCV.joblib',
        help='The path of segmentation output data')
        
        parser.add_argument(
        '-mri_input',
        type=str,
        default='',
        help='The path of segmentation output data')        

        parser.add_argument(
        '-mri_output',
        type=str,
        default='',
        help='The path of segmentation output data')        
        opt = parser.parse_args()

        return opt

    def get_args(self):
        print(self.args)
        return self.args    
  
    def inference(self):
        
        self.config.nii_to_image()    
        self.config.resize_images()

        # breakpoint()
        self.seg.customized_validation_sam()

        self.visual.process_folders()
        
        self.config.image_to_nii()
        self.config.mask_to_nii()

        _,_,collected_info=self.dose.build_dataset_with_validation()

        self.mri.inference()
        
        self.config.save(collected_info)


if __name__ == '__main__':
    # print("[DEBUG] args passed:", sys.argv)

    planner=Planner()
    print(planner.args)
    
    # breakpoint()
    planner.inference()

# python code/main.py  -niigz_path ./dataset  -prompt autonomy -seg_model ./seg_model/autonomy_v1.pth -dose_model ./dose_model/dose_model_BayesSearchCV.joblib  
