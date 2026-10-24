import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 1. تحميل البيانات وتدريب نموذج سريع للرسم
df = pd.read_csv("yemen_zakat_beneficiaries_120k.csv")
df['housing_encoded'] = df['housing_type'].astype('category').cat.codes
df['prev_aid_encoded'] = df['previous_aid_received'].map({'نعم': 1, 'لا': 0})

features = [
    'family_size', 'children_count', 'elderly_or_disabled_count',
    'monthly_income', 'housing_encoded', 'chronic_illness_count',
    'accumulated_debts', 'prev_aid_encoded'
]
X = df[features]
y = df['eligibility_category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# 2. توليد وحفظ رسم "أهمية المتغيرات" (Feature Importance Plot)
plt.figure(figsize=(10, 6))
importances = model.feature_importances_
indices = np.argsort(importances)

plt.title('Feature Importance for Zakat Eligibility Model', fontsize=14, fontweight='bold')
plt.barh(range(len(indices)), importances[indices], color='teal', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices], fontsize=11)
plt.xlabel('Relative Importance Score', fontsize=12)
plt.tight_layout()

chart_file_1 = "feature_importance.png"
plt.savefig(chart_file_1, dpi=300)
print(f"تم حفظ الرسم الأول بنجاح باسم: {chart_file_1}") # تم تصحيح الطباعة هنا
plt.close()

# 3. توليد وحفظ رسم التفسير المؤسسي (SHAP Summary Plot)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test.head(500)) # عينة لتسريع الرسم

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test.head(500), feature_names=features, show=False)
plt.title('SHAP Explainability Analysis for Model Decisions', fontsize=14, fontweight='bold')
plt.tight_layout()

chart_file_2 = "shap_explanation.png"
plt.savefig(chart_file_2, dpi=300, bbox_inches='tight')
print(f"تم حفظ الرسم الثاني بنجاح باسم: {chart_file_2}") # تم تصحيح الطباعة هنا
plt.close()