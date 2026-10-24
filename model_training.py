import pandas as pd
import numpy as np
import time
import shap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from rapidfuzz import fuzz

start_time = time.time()

# 1. تحميل قاعدة البيانات الضخمة (120 ألف سجل)
file_name = "yemen_zakat_beneficiaries_120k.csv"
print(f"جاري تحميل البيانات من الملف: {file_name}...")
df = pd.read_csv(file_name)

# 2. تجهيز وتحويل المتغيرات لنموذج التعلم الآلي (Preprocessing)
df['housing_encoded'] = df['housing_type'].astype('category').cat.codes
df['prev_aid_encoded'] = df['previous_aid_received'].map({'نعم': 1, 'لا': 0})

# تحديد الخصائص المدخلة (Features) والهدف (Target)
features = [
    'family_size', 'children_count', 'elderly_or_disabled_count',
    'monthly_income', 'housing_encoded', 'chronic_illness_count',
    'accumulated_debts', 'prev_aid_encoded'
]
target = 'eligibility_category'

X = df[features]
y = df[target]

# تقسيم البيانات إلى مجموعة تدريب (80%) ومجموعة اختبار (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. بناء وتدريب نموذج التعلم الآلي (Random Forest Classifier)
print("\nجاري تدريب نموذج تعلم الآلة (Random Forest)...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# 4. تقييم أداء نموذج التعلم الآلي
print("جاري تقييم دقة النموذج على بيانات الاختبار...")
y_pred = model.predict(X_test)
print(f"\nدقة النموذج (Accuracy): {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
print("تقرير الأداء التفصيلي (Classification Report):")
print(classification_report(y_test, y_pred))


# 5. نظام كشف الازدواجية (Duplicate Detection)
def check_duplicate(new_name, new_phone, existing_df, name_threshold=85):
    phone_match = existing_df[existing_df['phone_number'].astype(str) == str(new_phone)]
    if not phone_match.empty:
        return True, "تنبيه: رقم الهاتف مسجل مسبقاً في النظام!"

    for idx, row in existing_df.head(10000).iterrows():
        if fuzz.ratio(new_name, row['full_name']) >= name_threshold:
            return True, f"تنبيه: يوجد تشابه في الاسم مع مسجل سابق ({row['full_name']})"

    return False, "الاسم ورقم الهاتف غير مسجلين مسبقاً."


# 6. تفسير قرارات نموذج الـ ML مؤسسياً باستخدام SHAP
print("\nجاري إعداد محرك التفسير المؤسسي (SHAP Explainer) لشفافية نموذج الـ ML...")
explainer = shap.TreeExplainer(model)


def explain_ml_decision(applicant_data, feature_names):
    prediction = model.predict(applicant_data)[0]
    shap_values = explainer.shap_values(applicant_data)

    print(f"\n--- تقرير تفسير نموذج الـ ML لقرار الهيئة ---")
    print(f"تصنيف نموذج تعلم الآلة: [ {prediction} ]")
    print("أهم العوامل التي دفعته لهذا القرار:")

    class_idx = list(model.classes_).index(prediction)

    if isinstance(shap_values, list):
        vals = shap_values[class_idx][0]
    elif len(shap_values.shape) == 3:
        vals = shap_values[0, :, class_idx]
    else:
        vals = shap_values[0]

    vals = np.array(vals).flatten()
    sorted_factors = sorted(zip(feature_names, vals, applicant_data.values[0]), key=lambda x: abs(float(x[1])),
                            reverse=True)

    for feat, impact, val in sorted_factors[:4]:
        direction = "دفع نحو الاستحقاق العاجل" if impact > 0 else "خفف من الأولوية"
        print(f"  • المتغير ({feat}) بقيمته ({val}): تأثيره الرياضي كان ({direction})")


# اختبار حالة افتراضية جديدة للتفسير
sample_case = pd.DataFrame([[
    8,  # family_size
    5,  # children_count
    1,  # elderly_or_disabled_count
    0,  # monthly_income
    2,  # housing_encoded
    2,  # chronic_illness_count
    250000,  # accumulated_debts
    0  # prev_aid_encoded
]], columns=features)

explain_ml_decision(sample_case, features)
print(f"\nإجمالي الوقت المستغرق: {time.time() - start_time:.2f} ثانية.")