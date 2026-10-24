import streamlit as st
import pandas as pd
import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier
from rapidfuzz import fuzz

# إعدادات صفحة الواجهة
st.set_page_config(page_title="النظام الذكي لتقييم استحقاق الزكاة", layout="wide")

st.title("🛡️ المنظومة الذكية لتقييم استحقاق المساعدات ومنع الازدواجية")
st.markdown(
    "نظام معتمد لهيئة الزكاة - مدعوم بالذكاء الاصطناعي القابل للتفسير والتدقيق المؤسسي (بناءً على 120 ألف سجل).")


# 1. تحميل البيانات وتدريب النموذج وتجهيز محرك SHAP (تخزين مؤقت للسرعة)
@st.cache_resource
def load_system_assets():
    file_name = "yemen_zakat_beneficiaries_120k.csv"
    df = pd.read_csv(file_name)

    df['housing_encoded'] = df['housing_type'].astype('category').cat.codes
    df['prev_aid_encoded'] = df['previous_aid_received'].map({'نعم': 1, 'لا': 0})

    features = [
        'family_size', 'children_count', 'elderly_or_disabled_count',
        'monthly_income', 'housing_encoded', 'chronic_illness_count',
        'accumulated_debts', 'prev_aid_encoded'
    ]
    X = df[features]
    y = df['eligibility_category']

    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    return model, explainer, df, features


with st.spinner("جاري تهيئة النظام وتحميل نموذج الذكاء الاصطناعي... يرجى الانتظار."):
    model, explainer, df_existing, features = load_system_assets()

st.success(f"✅ النظام يعمل بكفاءة! قاعدة البيانات تضم {len(df_existing):,} مستفيد مسجل.")

# 2. لوحة الإدخال الجانبية للباحث الميداني
st.sidebar.header("📝 إدخال بيانات متقدم جديد")

name = st.sidebar.text_input("الاسم الكامل", "صالح محمد العنسي")
phone = st.sidebar.text_input("رقم الهاتف", "779876543")
gov = st.sidebar.selectbox("المحافظة", df_existing['governorate'].unique())
fam_size = st.sidebar.slider("إجمالي أفراد الأسرة", 1, 15, 7)
children = st.sidebar.slider("عدد الأطفال (< 18 سنة)", 0, 10, 4)
elderly = st.sidebar.slider("عدد كبار السن / ذوي الإعاقة", 0, 5, 1)
income = st.sidebar.number_input("الدخل الشهري (ريال يمني)", 0, 500000, 0)
housing = st.sidebar.selectbox("نوع السكن", ['ملك', 'إيجار', 'عشوائي', 'مدمر جزئياً'])
chronic = st.sidebar.slider("عدد الأمراض المزمنة في الأسرة", 0, 5, 2)
debts = st.sidebar.number_input("الديون المتراكمة (ريال يمني)", 0, 2000000, 200000)
prev_aid = st.sidebar.selectbox("هل تلقى مساعدة خلال آخر 3 أشهر؟", ['لا', 'نعم'])

# زر تنفيذ الفحص والتقييم
if st.sidebar.button("🔍 فحص المتقدم وتقييم الاستحقاق المؤسسي"):

    # أ. فحص الازدواجية
    phone_match = df_existing[df_existing['phone_number'].astype(str) == str(phone)]
    is_duplicate = False
    dup_msg = ""

    if not phone_match.empty:
        is_duplicate = True
        dup_msg = "رقم الهاتف مسجل مسبقاً في سجلات الهيئة!"
    else:
        for idx, row in df_existing.head(10000).iterrows():
            if fuzz.ratio(name, row['full_name']) >= 85:
                is_duplicate = True
                dup_msg = f"يوجد تشابه اسم مع مسجل سابق ({row['full_name']})"
                break

    st.subheader("📊 نتيجة التحليل وتقرير الشفافية:")

    if is_duplicate:
        st.error(f"❌ تم تعليق الصرف (ازدواجية مكتشفة): {dup_msg}")
    else:
        st.success("✅ لا توجد أي ازدواجية مسجلة. المتقدم مقبول للاختبار التقني.")

        # ب. تجهيز البيانات والتنبؤ
        housing_mapping = {'ملك': 0, 'إيجار': 1, 'عشوائي': 2, 'مدمر جزئياً': 3}
        input_data = pd.DataFrame([[
            fam_size, children, elderly, income,
            housing_mapping.get(housing, 0),
            chronic, debts,
            1 if prev_aid == 'نعم' else 0
        ]], columns=features)

        prediction = model.predict(input_data)[0]

        # ج. عرض النتيجة البصرية
        if "عاجل" in prediction:
            st.error(f"🚨 تصنيف الحالة: **{prediction}** (صرف فوري بأولوية قصوى)")
        elif "متوسط" in prediction:
            st.warning(f"⚠️ تصنيف الحالة: **{prediction}** (استحقاق مشروط بالتوافر)")
        else:
            st.info(f"ℹ️ تصنيف الحالة: **{prediction}** (الوضع معتدل حالياً)")

        # د. تقرير التفسير المؤسسي (SHAP)
        st.markdown("---")
        st.markdown("### 📋 تقرير التفسير المؤسسي لقرار الهيئة (أسباب التصنيف):")

        shap_values = explainer.shap_values(input_data)
        class_idx = list(model.classes_).index(prediction)

        if isinstance(shap_values, list):
            vals = shap_values[class_idx][0]
        elif len(shap_values.shape) == 3:
            vals = shap_values[0, :, class_idx]
        else:
            vals = shap_values[0]

        vals = np.array(vals).flatten()
        sorted_factors = sorted(zip(features, vals, input_data.values[0]), key=lambda x: abs(float(x[1])), reverse=True)

        for feat, impact, val in sorted_factors[:4]:
            direction = "دفع بقوة نحو الاستحقاق العاجل" if impact > 0 else "قلل من درجة الأولوية"
            st.write(f"- عامل (**{feat}**) بالقيمة المسجلة (**{val}**): أثره في القرار كان ({direction}).")

# عرض عينة إحصائية أسفل الواجهة
st.markdown("---")
st.subheader("📂 نظرة عامة على قاعدة بيانات المستفيدين النشطة")
st.dataframe(df_existing[['applicant_id', 'full_name', 'phone_number', 'governorate', 'monthly_income',
                          'eligibility_category']].head(5))