import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# НАСТРОЙКА СТРАНИЦЫ STREAMLIT
st.set_page_config(
    page_title="Ames Housing Price Predictor",
    page_icon="🏠",
    layout="wide"
)


# ЗАГРУЗКА АРТЕФАКТОВ МОДЕЛИ 
@st.cache_resource
def load_models():
    with open("house_prices_v2.pkl", "rb") as f:
        bundle = pickle.load(f)
    return bundle


try:
    artifacts = load_models()
    
    # Извлекаем компоненты препроцессинга точно по сохраненным ключам
    scaler = artifacts["scaler"]
    neighborhood_means = artifacts["neighborhood_means"]
    global_mean = artifacts["global_mean"]
    high_skew_features = artifacts["high_skew_features"]
    columns_layout = artifacts["columns_layout"]
    
    # Извлекаем обученные модели ансамбля
    model_xgb = artifacts["model_xgb"]
    model_lgb = artifacts["model_lgb"]
    model_ridge = artifacts["model_ridge"]

except FileNotFoundError:
    st.error("Ошибка: Файл 'house_prices_v2.pkl' не найден в директории приложения!")
    st.stop()
except KeyError as e:
    st.error(f"Ошибка ключа в pkl-файле: {e}. Проверьте структуру экспортируемого словаря.")
    st.stop()


# ФУНКЦИЯ ЭМУЛЯЦИИ ПАЙПЛАЙНА ДЛЯ СЫРЫХ ДАННЫХ 
def preprocess_input_data(df):
    all_data = df.copy()
    all_data.drop(["Utilities", "Street"], axis=1, inplace=True, errors="ignore")
    
    # 1. Заполнение None категорий
    none_cols = [
        "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu", "GarageType",
        "GarageFinish", "GarageQual", "GarageCond", "BsmtQual", "BsmtCond",
        "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "MasVnrType"
    ]
    for col in none_cols:
        if col in all_data.columns:
            all_data[col] = all_data[col].astype("object").fillna("None")
            
    # 2. Восстановление LotFrontage по зафиксированному на Train словарю районов
    if "LotFrontage" in all_data.columns and "Neighborhood" in all_data.columns:
        all_data["LotFrontage"] = all_data.apply(
            lambda row: neighborhood_means.get(row["Neighborhood"], global_mean) 
            if pd.isnull(row["LotFrontage"]) else row["LotFrontage"], axis=1
        )
        
    if "GarageYrBlt" in all_data.columns and "YearBuilt" in all_data.columns:
        all_data["GarageYrBlt"] = all_data["GarageYrBlt"].fillna(all_data["YearBuilt"])
        
    for col in ["MSSubClass", "MoSold", "YrSold"]:
        if col in all_data.columns:
            all_data[col] = all_data[col].astype(str)
            
    # Заполнение пропусков медианой/модой
    for col in all_data.select_dtypes(include=[np.number]).columns:
        all_data[col] = all_data[col].fillna(0)
    for col in all_data.select_dtypes(include=["object", "string"]).columns:
        all_data[col] = all_data[col].fillna("None")

    # 3. Фичи инжиниринг
    quality_map = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0}
    qual_cols = [
        "ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC", 
        "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond"
    ]
    for col in qual_cols:
        if col in all_data.columns:
            all_data[col] = all_data[col].map(quality_map).fillna(0)
            
    if "BsmtExposure" in all_data.columns:
        all_data["BsmtExposure"] = all_data["BsmtExposure"].map(
            {"Gd": 4, "Av": 3, "Mn": 2, "No": 1, "None": 0}
        ).fillna(0)
        
    all_data["TotalSF"] = all_data["1stFlrSF"] + all_data["2ndFlrSF"] + all_data["TotalBsmtSF"]
    all_data["TotalBath"] = all_data["FullBath"] + 0.5 * all_data["HalfBath"] + all_data["BsmtFullBath"] + 0.5 * all_data["BsmtHalfBath"]
    all_data["HouseAge"] = all_data["YrSold"].astype(int) - all_data["YearBuilt"]
    all_data["YearsSinceRemod"] = all_data["YrSold"].astype(int) - all_data["YearRemodAdd"]
    all_data["GarageAge"] = all_data["YrSold"].astype(int) - all_data["GarageYrBlt"]
    
    all_data["Total_Qual"] = all_data["TotalSF"] * all_data["OverallQual"]
    all_data["High_Quality_SF"] = all_data["1stFlrSF"] + all_data["2ndFlrSF"]
    all_data["Total_Porch"] = all_data["OpenPorchSF"] + all_data["EnclosedPorch"] + all_data["3SsnPorch"] + all_data["ScreenPorch"]
    all_data["Neighborhood_Encoded"] = all_data["Neighborhood"].map(neighborhood_means).fillna(global_mean)

    # 4. Исправление скоса по списку признаков из Train
    for feat in high_skew_features:
        if feat in all_data.columns:
            all_data[feat] = np.log1p(all_data[feat])
            
    # OHE кодирование
    all_data_encoded = pd.get_dummies(all_data, drop_first=True, dtype=int)
    
    # Защитное выравнивание структуры колонок (чтобы избежать ValueError)
    final_df = pd.DataFrame(0, index=all_data_encoded.index, columns=columns_layout)
    for col in columns_layout:
        if col in all_data_encoded.columns:
            final_df[col] = all_data_encoded[col]
            
    return final_df


# БОКОВОЕ МЕНЮ НАВИГАЦИИ
st.sidebar.title("Навигация")
page = st.sidebar.radio("Выберите раздел:", ["🚀 Автоматический Предикт", "📊 Итоги исследования"])


# РАЗДЕЛ АВТОМАТИЧЕСКОГО ИНФЕРЕНСА
if page == "🚀 Автоматический Предикт":
    st.title("🏠 Автоматическая оценка стоимости недвижимости")
    st.markdown("Загрузите файл формата `test.csv` для автоматического расчета предсказаний ансамблем моделей.")
    
    uploaded_file = st.file_uploader("Выберите CSV-файл для анализа", type=["csv"])
    
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
        st.success(f"Файл успешно загружен! Обнаружено объектов: {len(input_df)}")
        
        with st.spinner("Препроцессинг и инференс ансамбля"):
            # Прогоняем через логику второго дня
            X_encoded = preprocess_input_data(input_df)
            X_scaled = scaler.transform(X_encoded)
            
            # Генерация предсказаний
            preds_xgb = np.expm1(model_xgb.predict(X_encoded))
            preds_lgb = np.expm1(model_lgb.predict(X_encoded))
            preds_ridge = np.expm1(model_ridge.predict(X_scaled))
            
            # Блендинг на основе весов
            final_preds = (preds_xgb * 0.45) + (preds_lgb * 0.40) + (preds_ridge * 0.15)
            
            result_df = pd.DataFrame({
                "Id": input_df["Id"] if "Id" in input_df.columns else range(1, len(final_preds) + 1),
                "Predicted_SalePrice": np.round(final_preds, 2)
            })
            
        st.subheader("📊 Результаты прогнозирования")
        st.dataframe(result_df)
        
        csv_buffer = result_df.rename(columns={"Predicted_SalePrice": "SalePrice"}).to_csv(index=False)
        st.download_button(
            label="📥 Скачать файл предсказаний (Kaggle Format)",
            data=csv_buffer,
            file_name="final_submission.csv",
            mime="text/csv"
        )


# --- РАЗДЕЛ АНАЛИТИЧЕСКОЙ ПРЕЗЕНТАЦИИ ---
elif page == "📊 Итоги исследования":
    st.title("📈 Итоги исследования и архитектура ML-пайплайна")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Базовый скор", value="0.1373")
    with col2:
        st.metric(label="Лучший CV RMSLE (Ridge)", value="0.1146", delta="-0.0227", delta_color="inverse")
    with col3:
        st.metric(label="Итоговый соревновательный ансамбль", value="Топ~10%")
        
    st.subheader("📋 Таблица результатов кросс-валидации")
    metrics_data = {
        "Модель": ["Baseline XGBoost (сток)", "XGBoost (улучшенный)", "LightGBM", "Ridge Регрессия (Топ)"],
        "CV RMSLE": [0.1373, 0.1168, 0.1191, 0.1146],
        "R2 Score (Приблизительный)": [0.865, 0.911, 0.902, 0.923]
    }
    st.table(pd.DataFrame(metrics_data))
    
    st.subheader("Оценка важности признаков")
    feature_names = ["TotalSF", "Total_Qual", "OverallQual", "Neighborhood_Encoded", "HouseAge", "KitchenQual", "ExterQual", "High_Quality_SF"]
    feature_importance = [0.39, 0.28, 0.21, 0.19, -0.15, 0.13, 0.11, 0.08]
    
    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.barplot(x=feature_importance, y=feature_names, palette="coolwarm", ax=ax, hue=feature_names, legend=False)
    ax.set_title("Влияние ключевых признаков на логарифм стоимости (Ridge)")
    st.pyplot(fig)