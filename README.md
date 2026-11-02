# 상품별 판매 수요 예측 Streamlit 앱

UCI Machine Learning Repository의 **Online Retail II** 실제 거래 자료를 사용해 상품별 주간 판매량을 예측하는 학습용 프로젝트입니다. 화면의 상품 목록은 예시 상품명이 아니라 원본 거래 자료의 상품 설명과 상품 코드를 사용합니다.

## 데이터 출처

- 출처: [UCI Machine Learning Repository — Online Retail II](https://archive.ics.uci.edu/dataset/502/online%2Bretail%2Bii)
- 제공 자료: 2009년 12월부터 2011년 12월까지의 영국 온라인 소매 거래, 약 107만 건
- 주요 항목: 거래번호, 상품 코드, 상품 설명, 수량, 거래일시, 단가, 고객 번호, 국가
- 라이선스: CC BY 4.0
- 인용: Chen, D. (2012). *Online Retail II* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CG6D.

앱은 첫 실행 때 UCI에서 원본 ZIP 파일을 자동으로 가져옵니다. UCI 접속이 안 되면 원본 Excel 파일을 내려받아 앱 왼쪽에서 직접 올릴 수도 있습니다. 원본 데이터 파일은 GitHub에 올리지 않습니다.

## 앱 기능

- 실제 거래에 기록된 상품명과 상품 코드 선택
- 상품별 주간 판매 수량 예측 (Random Forest)
- 최근 판매량 그래프와 향후 1~8주 예측 보기
- 마지막 기간을 별도 평가해 MAE 확인
- 최근 주 판매량을 그대로 사용하는 기준 방법과 모델 성능 비교
- 해당 상품의 취소 수량을 별도 참고 정보로 확인

## 로컬에서 실행하기

Python 3.10 이상을 준비하고 프로젝트 폴더에서 아래 명령을 실행합니다.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

macOS/Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## GitHub와 Streamlit Community Cloud에 배포하기

1. 이 폴더의 파일을 GitHub 저장소에 올립니다. `online_retail_II.xlsx`나 ZIP 데이터 파일은 올리지 않습니다.
2. [Streamlit Community Cloud](https://share.streamlit.io/)에 GitHub 계정으로 로그인합니다.
3. **Create app**을 선택하고 저장소, 브랜치, 진입 파일 `app.py`를 지정한 다음 **Deploy**를 누릅니다.
4. 첫 실행은 UCI에서 데이터를 내려받으므로 준비가 끝날 때까지 잠시 기다립니다.

자세한 배포 안내는 [Streamlit 공식 문서](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app)를 참고하세요.

## 결과 해석 시 주의

- 예측 대상은 거래 자료에 기록된 판매 수량입니다. 품절 때문에 판매하지 못한 잠재 수요까지 포함하지 않습니다.
- 취소 수량은 판매량 예측과 분리해서 표시합니다. 주문과 취소가 항상 1:1로 연결된다고 가정하지 않습니다.
- 자료는 2009~2011년 영국 소매업체 기록이므로 오늘날 한국 온라인 쇼핑몰 판매량을 그대로 대표하지 않습니다.
- 평가 MAE는 이 자료에서 학습에 쓰지 않은 기간의 오차입니다. 실제 운영 성능을 보장하지 않습니다.
