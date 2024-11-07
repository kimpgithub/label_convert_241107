# label_convert_241107
yolo seg 학습을 위한 토지피복지도 항공위성 이미지(AIhub) 데이터의 라벨 가공

#. 환경 구성
- conda create -n yolo python=3.10
- pip install ultralytics
- pip install (설치 필요 패키지)
---
#. 파일 준비
- 파일 다운로드 : https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=71361 주소에서 데이 다운로드
- 압축 해제 : Training/01.원천데이터/TS_AP25_512픽셀.zip, Training/02.라벨링데이터/TL_AP25_512픽셀_AP25_512픽셀_Meta.zip, Training/02.라벨링데이터/TL_AP25_512픽셀_AP25_512픽셀_Json.zip
- 폴더명 변경 : TS_AP25_512, TS_AP25_512_META, TS_AP25_512_Json
---
- 변환 라벨링 시각화
- ![output_polygon_overlay (1)](https://github.com/user-attachments/assets/07351b82-4fee-490a-99dd-73b7dcc129a4)
