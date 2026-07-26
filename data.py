from opendartreader import OpenDartReader
import os, re

API_KEY = "d9b08ae9e2a5e2464311f2b55129345af9ba2b73".strip()
dart = OpenDartReader(API_KEY)

corp_name = "005380"  # 현대자동차 종목코드

reports = dart.list(corp_name, start="20220101", end="20260401", kind="A")

# "사업보고서"가 이름에 포함된 것만 (정정본도 포함 — 내용은 정상적인 전체 보고서라 그대로 써도 됨)
biz_reports = reports[reports["report_nm"].str.contains("사업보고서")].copy()

# report_nm에서 실제 사업연도 추출 (예: "사업보고서 (2023.12)" → "2023", "[기재정정]사업보고서 (2021.12)" → "2021")
def extract_year(name):
    m = re.search(r"\((\d{4})\.", name)
    return m.group(1) if m else None

biz_reports["fiscal_year"] = biz_reports["report_nm"].apply(extract_year)
biz_reports = biz_reports.dropna(subset=["fiscal_year"])

# 같은 사업연도에 여러 건(원본+정정)이 있으면, 제출일(rcept_dt)이 가장 최신인 것만 남기기
biz_reports = biz_reports.sort_values("rcept_dt").drop_duplicates(subset="fiscal_year", keep="last")

os.makedirs("reports", exist_ok=True)

for _, row in biz_reports.iterrows():
    fiscal_year = row["fiscal_year"]
    rcept_no = row["rcept_no"]
    try:
        text = dart.document(rcept_no)
        with open(f"reports/{corp_name}_{fiscal_year}.txt", "w", encoding="utf-8") as f:
            f.write(text)
        print(f"{fiscal_year}년 사업보고서 저장 완료 (report_nm={row['report_nm']})")
    except Exception as e:
        print(f"{fiscal_year}년 실패: {e}")