import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.metrics import mean_absolute_error, r2_score
from tqdm import tqdm  # 添加进度条
import numpy as np
import nibabel as nib
from skimage.measure import regionprops
import joblib,os

font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
my_font = fm.FontProperties(fname=font_path)

class DosePredictor:
    def __init__(self, args):
        self.args = args
        self.model_path = args.dose_model 
        self.csv_path = args.dose_csv_path  

    def extract_mri_features(self,mri_path, origin_mask):
        # 数据加载与校验
        mri_img = nib.load(mri_path)
        mask_img = nib.load(origin_mask)

        # 维度匹配检查
        if mri_img.shape != mask_img.shape:
            raise ValueError("MRI与掩膜维度不匹配")

        # 自动选择有效切片
        mask_data = mask_img.get_fdata()#.squeeze()
        # breakpoint()
        valid_slices = np.where(np.any(mask_data > 0, axis=(0, 1)))[0]

        # 处理空掩膜情况
        if len(valid_slices) == 0:
            raise ValueError("未找到有效掩膜区域")

        # 智能切片选择逻辑
        slice_idx = valid_slices[len(valid_slices) // 2]  # 取中间有效层


        # 特征提取
        # plt.figure()
        mri_slice = mri_img.get_fdata()[:, :, slice_idx].T
        mask_slice = mask_data[:, :, slice_idx].T.astype(bool)
        masked_slice = np.where(mask_slice, mri_slice, np.nan)
        # plt.imshow(masked_slice)
        # plt.show()

        # 二值化处理（假设ROI已标注）
        mask = (mask_slice > 0).astype(np.uint8)
        props = regionprops(mask)

        
        features = {
            # 几何特征
            'volume': props[0].area * np.prod(mri_img.header.get_zooms()),
            'equivalent_diameter': props[0].equivalent_diameter,
            'eccentricity': props[0].eccentricity,

            # 灰度特征
            'mean_intensity': np.mean(masked_slice[mask == 1]),
            'intensity_std': np.std(masked_slice[mask == 1]),

            # 纹理特征
            'hu_moments': props[0].moments_hu,
        }
        # breakpoint()
        return features,slice_idx

    def build_dataset_with_validation(self):
        """
        增强版数据构建函数，包含预测验证功能
        输入：
            csv_path - 包含患者ID和真实剂量的CSV路径
            model_path - 训练好的模型路径
        输出：
            包含预测结果和评估指标的DataFrame
        """
        # 加载模型和特征
        args = self.args
        model_path = self.model_path

        model_data = joblib.load(model_path)
        pipeline = model_data['pipeline']
        feature_names = model_data['feature_names']

        # 读取原始数据
        # df = pd.read_csv(csv_path)

        # 结果容器
        results = []
        failed_cases = []
        collected_info = []

        patient_ids = os.listdir(args.image_path)
        for patient_id in tqdm(patient_ids, total=len(patient_ids), desc="Processing Patients"):
            try:
                ori_file = os.path.join(args.image_nii_path, f"{patient_id}.nii.gz")
                msk_file = os.path.join(args.mask_nii_path, f"{patient_id}.nii.gz")

                # 特征提取
                features, slice_idx = self.extract_mri_features(ori_file, msk_file)

                # 构建预测输入
                hu_moments = features.pop('hu_moments')
                feature_dict = features.copy()
                for i, val in enumerate(hu_moments):
                    feature_dict[f"hu_{i}"] = val

                X_new = pd.DataFrame([feature_dict], columns=feature_names)

                predicted_dose = int(pipeline.predict(X_new)[0])

                collected_info.append((patient_id, slice_idx, predicted_dose))
                results.append({
                    'patient_id': patient_id,
                    'predicted_dose': predicted_dose,
                    'slice_idx': slice_idx
                })

            except Exception as e:
                failed_cases.append({
                    'patient_id': patient_id,
                    'error': str(e)
                })
                print(f"Patient ID: {patient_id}, Error: {e}")

        result_df = pd.DataFrame(results)

        # 可选：保存结果
        # result_df.to_csv("predicted_dose_only.csv", index=False)

        # plt.show()

        return result_df, failed_cases,collected_info

    def inference(self):
        # 这里可以添加调用 dose_predict 的代码
        result_df, errors,collected_info = self.build_dataset_with_validation()

        # 保存结果
        result_df.to_csv("/root/autodl-tmp/gbli/ultrasound_dose_prediction/validation_results.csv", index=False)
        for fail in errors:
           print(errors)
           raise RuntimeError(f"Patient ID: {fail['patient_id']}, Error: {fail['error']}")
            
        print("save result done")
        return collected_info

# 使用示例
if __name__ == "__main__":
    # # 生成验证结果
    # csv_path = "./data/total_id_doses1.csv"
    # model_path = "dose_model_BayesSearchCV.joblib"
    # result_df, errors = build_dataset_with_validation(csv_path, model_path)

    # # 保存结果
    # result_df.to_csv("validation_results.csv", index=False)
    # for fail in errors:
    #     print(f"Patient ID: {fail['patient_id']}, Error: {fail['error']}")
    # print("save result done")
    pass