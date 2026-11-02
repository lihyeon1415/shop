from __future__ import annotations

from io import BytesIO
from zipfile import ZipFile

import pandas as pd
import requests
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


UCI_DATA_URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
UCI_PAGE = "https://archive.ics.uci.edu/dataset/502/online%2Bretail%2Bii"
EXCLUDED_SERVICE_WORDS = "POSTAGE|BANK CHARGES|DOTCOM|AMAZON FEE|ADJUST BAD DEBT|CARRIAGE|MANUAL"
PRODUCT_NAMES_KO = {
    "85048": "15cm 크리스마스 유리 장식볼 조명 세트 (20구)",
    "79323P": "분홍 체리 장식 조명",
    "79323W": "흰색 체리 장식 조명",
    "22041": "7인치 싱글 레코드 액자",
    "21232": "딸기 모양 도자기 소품 보관함",
    "22064": "분홍 도넛 모양 소품 보관함",
    "21871": "환경 보호 머그컵",
    "21523": "홈 스위트 홈 현관 매트",
    "22350": "고양이 밥그릇",
    "85123A": "흰색 하트 걸이형 티라이트 홀더",
    "71053": "흰색 금속 랜턴",
    "84406B": "크림색 하트 옷걸이",
    "84029G": "영국 국기 무늬 뜨개 보온 물주머니",
    "84029E": "빨간색 하트 무늬 보온 물주머니",
    "22752": "마트료시카 인형 상자 7개 세트",
    "21730": "서리 무늬 유리 별 모양 촛대",
    "22633": "영국 국기 무늬 손난로",
    "22632": "빨간색 물방울무늬 손난로",
    "84879": "여러 색상 새 장식품",
    "85099B": "대형 빨간색 복고풍 점무늬 가방",
    "22423": "3단 케이크 스탠드",
    "47566": "파티 가랜드",
    "20725": "빨간색 복고풍 점무늬 도시락 가방",
    "22197": "팝콘 용기",
    "21212": "복고풍 점무늬 컵케이크 종이틀 72개 묶음",
    "23084": "토끼 모양 수면등",
    "22961": "잼 만들기 세트",
}


def korean_product_name(code: str, original_name: str) -> str:
    translated = PRODUCT_NAMES_KO.get(str(code).strip())
    if translated:
        return translated
    return f"상품 코드 {code} (한글 상품명 미등록)"

st.set_page_config(page_title="상품 판매 수요 예측", page_icon="📦", layout="wide")


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Accept the UCI workbook's original headers and common CSV variants."""
    aliases = {
        "invoice": "InvoiceNo",
        "invoiceno": "InvoiceNo",
        "stockcode": "StockCode",
        "description": "Description",
        "quantity": "Quantity",
        "invoicedate": "InvoiceDate",
        "price": "UnitPrice",
        "unitprice": "UnitPrice",
        "customerid": "CustomerID",
        "country": "Country",
    }
    rename = {}
    for column in frame.columns:
        key = str(column).strip().lower().replace(" ", "").replace("_", "")
        if key in aliases:
            rename[column] = aliases[key]
    frame = frame.rename(columns=rename)
    required = {"InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate", "UnitPrice"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError("필수 열이 없습니다: " + ", ".join(sorted(missing)))
    return frame


def read_uploaded_file(upload) -> pd.DataFrame:
    content = upload.getvalue()
    if upload.name.lower().endswith(".csv"):
        return normalize_columns(pd.read_csv(BytesIO(content)))
    sheets = pd.read_excel(
        BytesIO(content),
        sheet_name=None,
        engine="openpyxl",
        usecols=lambda column: str(column).strip().lower().replace(" ", "")
        in {"invoice", "invoiceno", "stockcode", "description", "quantity", "invoicedate", "price", "unitprice"},
    )
    return normalize_columns(pd.concat(sheets.values(), ignore_index=True))


@st.cache_data(show_spinner="UCI 거래 자료를 불러오고 정리하고 있어요. 첫 실행에는 잠시 걸릴 수 있어요.")
def load_uci_data() -> pd.DataFrame:
    response = requests.get(UCI_DATA_URL, timeout=180)
    response.raise_for_status()
    with ZipFile(BytesIO(response.content)) as archive:
        excel_name = next(name for name in archive.namelist() if name.lower().endswith(".xlsx"))
        excel_bytes = BytesIO(archive.read(excel_name))
    sheets = pd.read_excel(
        excel_bytes,
        sheet_name=None,
        engine="openpyxl",
        usecols=lambda column: str(column).strip().lower().replace(" ", "")
        in {"invoice", "invoiceno", "stockcode", "description", "quantity", "invoicedate", "price", "unitprice"},
    )
    return normalize_columns(pd.concat(sheets.values(), ignore_index=True))


def prepare_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = normalize_columns(raw.copy())
    data["InvoiceNo"] = data["InvoiceNo"].astype(str).str.strip()
    data["StockCode"] = data["StockCode"].astype(str).str.strip()
    data["Description"] = data["Description"].fillna("상품명 미기재").astype(str).str.strip()
    data["Quantity"] = pd.to_numeric(data["Quantity"], errors="coerce")
    data["UnitPrice"] = pd.to_numeric(data["UnitPrice"], errors="coerce")
    data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"], errors="coerce")
    data = data.dropna(subset=["StockCode", "Quantity", "UnitPrice", "InvoiceDate"])
    data = data[data["StockCode"].ne("")]

    cancel_mask = data["InvoiceNo"].str.upper().str.startswith("C") | data["Quantity"].lt(0)
    cancellations = data.loc[cancel_mask].copy()
    cancellations["CancelQuantity"] = cancellations["Quantity"].abs()
    cancellations["Week"] = cancellations["InvoiceDate"].dt.to_period("W-SUN").dt.start_time
    cancel_weekly = (
        cancellations.groupby(["StockCode", "Week"], as_index=False)["CancelQuantity"]
        .sum()
    )

    sales = data.loc[~cancel_mask & data["Quantity"].gt(0) & data["UnitPrice"].gt(0)].copy()
    service_mask = sales["Description"].str.upper().str.contains(EXCLUDED_SERVICE_WORDS, regex=True, na=False)
    sales = sales.loc[~service_mask]
    sales["Week"] = sales["InvoiceDate"].dt.to_period("W-SUN").dt.start_time
    weekly = sales.groupby(["StockCode", "Week"], as_index=False).agg(
        SalesQuantity=("Quantity", "sum"),
        Description=("Description", lambda values: values.mode().iloc[0] if not values.mode().empty else values.iloc[0]),
    )
    return weekly, cancel_weekly


def make_features(series: pd.Series) -> pd.DataFrame:
    frame = pd.DataFrame({"target": series.astype(float)})
    for lag in (1, 2, 4, 8):
        frame[f"lag_{lag}"] = frame["target"].shift(lag)
    frame["rolling_4"] = frame["target"].shift(1).rolling(4).mean()
    frame["week_of_year"] = frame.index.isocalendar().week.astype(int)
    frame["month"] = frame.index.month
    return frame.dropna()


def forecast_product(weekly: pd.DataFrame, product_code: str, horizon: int):
    product_rows = weekly.loc[weekly["StockCode"].eq(product_code)]
    if product_rows.empty:
        raise ValueError("선택한 상품의 판매 기록을 찾지 못했어요.")
    first_week = weekly["Week"].min()
    last_week = weekly["Week"].max()
    timeline = pd.date_range(first_week, last_week, freq="W-MON")
    series = product_rows.set_index("Week")["SalesQuantity"].groupby(level=0).sum().reindex(timeline, fill_value=0)
    features = make_features(series)
    if len(features) < 12:
        raise ValueError("이 상품의 기록이 부족해 예측을 만들 수 없어요. 다른 상품을 선택해 주세요.")

    holdout = min(8, max(2, len(features) // 5))
    train = features.iloc[:-holdout]
    test = features.iloc[-holdout:]
    columns = ["lag_1", "lag_2", "lag_4", "lag_8", "rolling_4", "week_of_year", "month"]
    model = RandomForestRegressor(n_estimators=120, max_depth=7, min_samples_leaf=2, random_state=42, n_jobs=-1)
    model.fit(train[columns], train["target"])
    test_prediction = model.predict(test[columns]).clip(0)
    model_mae = mean_absolute_error(test["target"], test_prediction)
    baseline_prediction = test["lag_1"].to_numpy()
    baseline_mae = mean_absolute_error(test["target"], baseline_prediction)

    final_model = RandomForestRegressor(n_estimators=120, max_depth=7, min_samples_leaf=2, random_state=42, n_jobs=-1)
    final_model.fit(features[columns], features["target"])
    future_values = list(series.astype(float).to_numpy())
    future_dates = pd.date_range(series.index[-1] + pd.Timedelta(days=7), periods=horizon, freq="W-MON")
    future_predictions = []
    for future_date in future_dates:
        row = {
            "lag_1": future_values[-1],
            "lag_2": future_values[-2],
            "lag_4": future_values[-4],
            "lag_8": future_values[-8],
            "rolling_4": sum(future_values[-4:]) / 4,
            "week_of_year": int(future_date.isocalendar().week),
            "month": future_date.month,
        }
        prediction = max(0.0, float(final_model.predict(pd.DataFrame([row])[columns])[0]))
        future_predictions.append(prediction)
        future_values.append(prediction)
    return series, test, test_prediction, baseline_mae, model_mae, future_dates, future_predictions


st.title("상품별 판매 수요 예측")
st.write("실제 온라인 소매 거래 기록을 불러와 상품별 주간 판매량을 학습하고, 다음 주 판매 수량을 예측합니다. 확인된 품목은 한글명으로 표시하고, 상품 코드로 원본과 대조할 수 있습니다.")
st.caption("예측 대상은 실제 판매 수량입니다. 재고 부족으로 판매하지 못한 수요까지 포함하는 값은 아닙니다.")

with st.sidebar:
    st.header("분석 설정")
    st.link_button("UCI 원본 데이터 출처", UCI_PAGE)
    st.caption("첫 접속에는 약 43.5MB 데이터 다운로드와 정리에 시간이 걸릴 수 있어요.")
    uploaded = st.file_uploader("또는 UCI 파일 직접 올리기", type=["xlsx", "csv"])

try:
    if uploaded is not None:
        raw_data = read_uploaded_file(uploaded)
        source_label = f"업로드 파일: {uploaded.name}"
    else:
        raw_data = load_uci_data()
        source_label = "UCI Online Retail II 원본 데이터"
    weekly_sales, weekly_cancellations = prepare_data(raw_data)
except Exception as exc:
    st.error("데이터를 불러오지 못했어요. 인터넷 연결을 확인하거나 UCI에서 내려받은 Excel 파일을 직접 올려주세요.")
    st.caption(f"자세한 내용: {exc}")
    st.stop()

st.caption(f"사용 데이터: {source_label} · 판매 기록 {len(weekly_sales):,}개 · 취소 기록 {len(weekly_cancellations):,}개")

product_totals = weekly_sales.groupby("StockCode")["SalesQuantity"].sum().sort_values(ascending=False)
product_codes = product_totals.head(30).index.tolist()
product_names = (
    weekly_sales.sort_values("Week").groupby("StockCode")["Description"].last().to_dict()
)
with st.sidebar:
    product_code = st.selectbox(
        "상품 선택 (거래 자료에 실제 등록된 상품)",
        product_codes,
        format_func=lambda code: f"{korean_product_name(code, product_names.get(code, ''))} · {code}",
    )
    horizon = st.slider("예측 기간 (주)", min_value=1, max_value=8, value=4)

product_name = korean_product_name(product_code, product_names.get(product_code, ""))
cancel_total = weekly_cancellations.loc[weekly_cancellations["StockCode"].eq(product_code), "CancelQuantity"].sum()

try:
    series, test, test_prediction, baseline_mae, model_mae, future_dates, future_predictions = forecast_product(
        weekly_sales, product_code, horizon
    )
except ValueError as exc:
    st.warning(str(exc))
    st.stop()

next_week = round(future_predictions[0])
average_error = round(model_mae, 1)
col1, col2, col3 = st.columns(3)
col1.metric("선택 상품", product_name, help=f"실제 거래 자료의 상품 코드: {product_code}")
col2.metric("다음 주 예상 판매량", f"{next_week:,}개")
col3.metric("평가 오차 (MAE)", f"{average_error:,}개", help="학습에 사용하지 않은 마지막 주들의 평균 절대 오차입니다. 낮을수록 예측 오차가 작습니다.")

st.subheader("주간 판매량과 예측")
history = series.tail(20).rename("실제 판매량").to_frame()
history["모델 예측 (평가 구간)"] = pd.Series(test_prediction, index=test.index).reindex(history.index)
future_frame = pd.DataFrame({"모델 예측 (미래)": future_predictions}, index=future_dates)
chart = pd.concat([history, future_frame], axis=0)
st.line_chart(chart, y_label="수량 (개)", x_label="주 시작일", height=360)

with st.expander("예측 성능과 취소 기록 보기"):
    st.write(f"최근 {len(test)}주를 평가용으로 남겨 두고, 그 이전 거래로 모델을 학습했어요.")
    comparison = pd.DataFrame(
        {"방법": ["최근 주 판매량 그대로 사용", "Random Forest 모델"], "MAE (개)": [round(baseline_mae, 2), round(model_mae, 2)]}
    )
    st.dataframe(comparison, hide_index=True, use_container_width=True)
    st.metric("이 상품의 전체 기록상 취소 수량", f"{cancel_total:,.0f}개")
    st.caption("취소 수량은 별도 참고값입니다. 주문과 취소가 항상 1:1로 짝지어진다고 가정하지 않습니다.")

st.caption("데이터: Chen, D. (2012), Online Retail II, UCI Machine Learning Repository, CC BY 4.0.")
