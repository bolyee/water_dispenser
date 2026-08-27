import os
import yt_dlp
from datetime import timedelta

video_url = "https://youtu.be/0P4JWcgotdI?si=cofm0_rex4KAbS7R"
total_count = 20  # 다운로드할 파일 총 개수
interval_minutes = 5  # 간격 (5분)
duration_minutes = 2  # 추출할 길이 (2분)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

# 시간 데이터를 HH:MM:SS 문자열로 변환해주는 헬퍼 함수
def to_time_str(minutes):
    td = timedelta(minutes=minutes)
    # td가 05:00:00 처럼 나올 수 있으므로 zfill로 포맷팅
    return str(td).zfill(8)

print("🚀 데이터셋 자동 다운로드를 시작합니다...")

for i in range(total_count):
    # 시작 시간과 종료 시간 계산 (분 단위)
    start_min = i * interval_minutes
    end_min = start_min + duration_minutes
    
    start_time = to_time_str(start_min)
    end_time = to_time_str(end_min)
    file_name = f"dataset_{str(i+1).zfill(2)}"  # dataset_01, dataset_02 ...
    
    print(f"\n[구간 {i+1}/{total_count}] 🔄 추출 중: {file_name} ({start_time} ~ {end_time})")
    
    ydl_opts = {
        'format': 'bestaudio/best',
        # DPAPI 복호화 오류 우회 및 인증을 위해 텍스트 쿠키 파일을 직접 주입합니다.
        'cookiefile': os.path.join(ROOT_DIR, 'cookies.txt'),
        'extractor_args': {
            'youtube': {
                'player_client': ['ios', 'android']  # 클라이언트를 모바일 앱으로 스푸핑하여 차단 우회
            }
        },
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'download_sections': [{
            'start_time': start_time,
            'end_time': end_time,
            'title': file_name
        }],
        'outtmpl': f"{os.path.join(ROOT_DIR, file_name)}.%(ext)s",
        'quiet': True,  # 터미널에 지저분한 로그 안 나오게 켜둠
        'noprogress': True
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
    except Exception as e:
        print(f"❌ {file_name} 다운로드 중 에러 발생: {e}")

print("\n✅ 총 20개의 데이터셋 오디오 추출이 완료되었습니다!")
