import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

# 假设已经预测并得到预测结果
y_true = [2, 0, 1, 3, 9, 5, 7, 8, 6, 4]  # 示例
y_pred_A = [2, 0, 1, 3, 9, 5, 7, 8, 6, 4]
y_pred_B = [2, 0, 1, 3, 9, 5, 7, 8, 6, 4]
y_pred_C = [2, 0, 1, 3, 9, 5, 7, 8, 6, 4]

cm_A = confusion_matrix(y_true, y_pred_A)
cm_B = confusion_matrix(y_true, y_pred_B)
cm_C = confusion_matrix(y_true, y_pred_C)

plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
sns.heatmap(cm_A, annot=True, fmt="d", cmap="YlGnBu")
plt.title('Confusion Matrix for Model A')
plt.xlabel('Predicted')
plt.ylabel('True')

plt.subplot(1, 3, 2)
sns.heatmap(cm_B, annot=True, fmt="d", cmap="YlGnBu")
plt.title('Confusion Matrix for Model B')
plt.xlabel('Predicted')
plt.ylabel('True')

plt.subplot(1, 3, 3)
sns.heatmap(cm_C, annot=True, fmt="d", cmap="YlGnBu")
plt.title('Confusion Matrix for Model C')
plt.xlabel('Predicted')
plt.ylabel('True')

plt.tight_layout()
plt.show()
