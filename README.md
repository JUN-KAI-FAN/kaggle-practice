# 🏥 乳房X光影像分類專案 (Mammography Classification)

## 📋 專案簡介

本專案使用YOLOv8模型進行乳房X光影像的二分類任務，目標是自動識別乳房腫瘤的良性（BENIGN）或惡性（MALIGNANT）。此專案採用了醫學影像專用的資料處理和評估方法，確保模型在醫療應用場景中的可靠性。

## 🎯 專案目標

- **二分類任務**: 自動分類乳房X光影像為良性或惡性
- **醫學影像優化**: 使用專門針對醫學影像設計的資料增強技術
- **高性能模型**: 達到約73.5%的驗證準確率和69.3%的測試準確率
- **臨床指標評估**: 提供敏感性、特異性等醫學診斷重要指標

## 📊 資料集統計

- **總資料量**: 3,568筆影像資料
- **訓練資料**: 2,864張影像 (經平衡處理後2,527張)
- **測試資料**: 704張影像
- **驗證資料**: 506張影像
- **類別分佈**:
  - 良性(BENIGN): 1,683張 (訓練集)
  - 惡性(MALIGNANT): 1,181張 (訓練集)

## 🔧 技術架構

### 模型架構
- **基礎模型**: YOLOv8s Classification
- **預訓練權重**: 使用ImageNet預訓練模型
- **影像大小**: 640x640像素
- **批次大小**: 16

### 資料增強策略
```python
基礎增強:
- 水平翻轉 (p=0.5)
- 小角度旋轉 (±10度)
- 隨機亮度對比度調整
- 高斯雜訊
- CLAHE直方圖均衡

強化增強 (用於少數類別):
- 更強的幾何變換
- 彈性變形
- 更多增強副本 (3x)
```

### 訓練配置
```yaml
epochs: 50
learning_rate: 0.001
optimizer: AdamW
scheduler: cosine annealing
device: CUDA (RTX 3060 Ti)
```

## 📈 模型性能

### 整體性能指標
- **準確率 (Accuracy)**: 69.3%
- **加權精確率 (Precision)**: 類別平衡的精確率
- **加權召回率 (Recall)**: 類別平衡的召回率
- **F1分數**: 綜合評估指標

### 醫學診斷指標
- **敏感性 (Sensitivity)**: 正確識別惡性腫瘤的能力
- **特異性 (Specificity)**: 正確識別良性腫瘤的能力
- **陽性預測值 (PPV)**: 預測為惡性時的準確性
- **陰性預測值 (NPV)**: 預測為良性時的準確性
- **AUC-ROC**: ROC曲線下面積

## 🗂️ 檔案結構

```
mammography_classification/
├── mammography_classification_improved.ipynb  # 主要訓練notebook
├── yolov8s_improved/
│   └── weights/
│       └── best.pt                           # 最佳訓練權重
├── yolo_cls_dataset/                         # YOLOv8格式資料集
│   ├── train/
│   │   ├── BENIGN/                          # 良性影像
│   │   └── MALIGNANT/                       # 惡性影像
│   ├── val/
│   └── test/
├── archive/                                  # 原始資料
│   ├── csv/                                 # 標籤資料
│   └── jpeg/                                # JPEG影像檔案
└── README.md                                # 專案說明文件
```

## 🚀 快速開始

### 環境需求
```bash
pip install ultralytics
pip install torch torchvision
pip install opencv-python
pip install albumentations
pip install scikit-learn
pip install pandas numpy matplotlib seaborn
```

### 使用訓練好的模型進行預測
```python
from ultralytics import YOLO

# 載入訓練好的模型
model = YOLO('mammography_classification/yolov8s_improved/weights/best.pt')

# 對單張影像進行預測
results = model.predict('path/to/mammography_image.jpg')

# 取得預測結果
for result in results:
    # 分類結果
    top1_idx = result.probs.top1
    confidence = result.probs.top1conf.item()
    predicted_label = result.names[top1_idx]
    print(f"預測類別: {predicted_label}, 信心度: {confidence:.3f}")
```

### 重新訓練模型
```python
from ultralytics import YOLO

# 載入預訓練模型
model = YOLO('yolov8s-cls.pt')

# 開始訓練
results = model.train(
    data='yolo_cls_dataset',
    epochs=50,
    imgsz=640,
    batch=16,
    lr0=0.001,
    device='cuda'
)
```

## 📋 資料預處理流程

1. **資料來源分析**: 從CSV檔案讀取影像路徑和標籤
2. **DICOM到JPEG對應**: 建立DICOM ID與JPEG檔案的對應關係
3. **Mask檔案過濾**: 自動識別並排除ROI mask檔案
4. **影像品質檢查**: 過濾掉損壞或過小的影像
5. **類別平衡**: 處理類別不平衡問題，採用下採樣和資料增強
6. **資料分割**: 訓練/驗證/測試集分割 (80%/20%/獨立測試集)

## 🔬 醫學影像特殊考量

### 資料增強限制
- 避免過度變形，保持醫學影像的診斷特徵
- 使用CLAHE增強對比度，提升病灶可見度
- 限制旋轉角度，符合臨床拍攝習慣

### 評估指標選擇
- 優先考慮敏感性（避免漏診惡性腫瘤）
- 平衡特異性（減少誤診良性為惡性）
- 提供ROC曲線用於閾值調整

### 臨床應用建議
- 建議作為輔助診斷工具，不應完全取代醫師判斷
- 需要定期使用新資料重新訓練模型
- 建議在不同醫院資料上進行外部驗證

## 📚 參考資料

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Albumentations Documentation](https://albumentations.ai/)
- Medical Image Classification Best Practices
- DICOM Medical Imaging Standards

## 👥 貢獻者

- 專案開發：[您的姓名]
- 資料處理：醫學影像專業團隊
- 模型優化：機器學習工程師

## 📄 授權

本專案僅供學術研究使用，不可用於商業醫療診斷。

## 🆘 支援

如有問題請提交Issue或聯繫專案維護者。

---

**⚠️ 重要提醒**: 本模型僅供研究用途，不應作為醫療診斷的唯一依據。任何醫療決策都應諮詢專業醫師。 