import requests
import time

ESP_IP = "192.168.0.250"
STOP_URL = f"http://{ESP_IP}/stop"

print("=" * 50)
print(" 🤖 ESP32 SG90 서보모터 통신 테스트 스크립트")
print("=" * 50)
print(f"대상 주소: {STOP_URL}\n")

input("▶ 준비되셨으면 Enter 키를 누르세요. (ESP32로 정지 신호를 쏩니다) ")

print("\n📡 ESP32로 신호 전송 중...")

try:
    start_time = time.time()
    # ESP32가 파이썬 요청을 튕겨내지 않도록 브라우저처럼 위장(User-Agent)하고 연결 유지 옵션을 끕니다.
    headers = {'Connection': 'close', 'User-Agent': 'Mozilla/5.0'}
    response = requests.get(STOP_URL, headers=headers, timeout=5)
    elapsed = time.time() - start_time
    
    print(f"\n✅ 통신 성공!")
    print(f"  - 응답 코드: {response.status_code}")
    print(f"  - 소요 시간: {elapsed:.2f}초")
    print(f"  - 응답 내용: {response.text.strip()}")
    
    print("\n🎉 서보모터가 정상적으로 작동했는지 하드웨어를 확인해 보세요!")
    
except requests.exceptions.Timeout:
    print("\n❌ 통신 실패: 응답 시간 초과 (Timeout)")
    print("  → 원인: ESP32 전원이 꺼져 있거나, PC와 같은 와이파이 망에 연결되지 않았을 수 있습니다.")
except requests.exceptions.ConnectionError as e:
    print("\n⚠️ 파이썬에서 에러가 떴지만, 서보모터는 작동했을 수도 있습니다!")
    print("  → ESP32가 명령은 실행했는데 파이썬에게 '완료' 응답을 보내기 전에 통신을 끊어버려서 발생하는 현상입니다.")
    print(f"  → 상세 에러: {e}")
    print("  → 해결: 만약 모터가 징~ 하고 돌아갔다면 이 에러는 무시하셔도 됩니다.")
except requests.exceptions.RequestException as e:
    print("\n❌ 통신 실패: 네트워크 에러")
    print(f"  → 상세 정보: {e}")
print("\n테스트 스크립트를 종료합니다.")
