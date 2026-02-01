"""
페르소나 프로필 이미지 생성 스크립트
OpenAI DALL-E API를 사용하여 AI 생성 이미지 제작
"""
import os
import httpx
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()


# 이미지 생성 프롬프트 정의
PERSONA_PROMPTS = {
    "facebook_un_michael": {
        "filename": "facebook_un_michael.jpg",
        "prompt": (
            "Professional portrait photo of a 45-year-old white American male doctor, "
            "wearing a blue UN vest over casual clothes, warm and trustworthy smile, "
            "humanitarian medical camp in background, natural lighting, "
            "realistic photograph style, no text or watermarks"
        ),
    },
    "kakao_jihye": {
        "filename": "kakao_jihye.jpg",
        "prompt": (
            "Professional portrait photo of a 31-year-old Korean businesswoman, "
            "elegant and sophisticated, wearing modern office attire, "
            "sitting in a luxury cafe with warm lighting, confident friendly smile, "
            "realistic photograph style, no text or watermarks"
        ),
    },
    "instagram_bella": {
        "filename": "instagram_bella.jpg",
        "prompt": (
            "Professional portrait photo of a 26-year-old Latina fashion model, "
            "glamorous and stylish, wearing a summer dress, "
            "Miami beach resort background with palm trees, bright natural lighting, "
            "Instagram influencer style, realistic photograph, no text or watermarks"
        ),
    },
    "x_crypto_alex": {
        "filename": "x_crypto_alex.jpg",
        "prompt": (
            "Professional portrait photo of a 32-year-old Asian man, "
            "casual tech entrepreneur style, wearing a designer t-shirt, "
            "Dubai skyline visible through window in background, confident expression, "
            "modern minimalist setting, realistic photograph style, no text or watermarks"
        ),
    },
    "telegram_prosecutor": {
        "filename": "telegram_prosecutor.jpg",
        "prompt": (
            "Professional portrait photo of a 38-year-old Korean man, "
            "wearing a formal dark suit and tie, serious authoritative expression, "
            "government office building interior background, "
            "formal ID photo style, realistic photograph, no text or watermarks"
        ),
    },
    "line_yuki": {
        "filename": "line_yuki.jpg",
        "prompt": (
            "Professional portrait photo of a 29-year-old Korean woman, "
            "neat casual office style, friendly warm smile, "
            "Tokyo Shibuya street or modern office background, "
            "natural daylight, realistic photograph style, no text or watermarks"
        ),
    },
}


def generate_and_save_images(output_dir: str = "assets/personas"):
    """DALL-E를 사용하여 페르소나 이미지 생성 및 저장"""

    # OpenAI 클라이언트 초기화
    client = OpenAI()  # OPENAI_API_KEY 환경변수 사용

    # 출력 디렉토리 생성
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"이미지 저장 경로: {output_path.absolute()}")
    print(f"총 {len(PERSONA_PROMPTS)}개 이미지 생성 시작...\n")

    for persona_id, config in PERSONA_PROMPTS.items():
        filename = config["filename"]
        prompt = config["prompt"]
        filepath = output_path / filename

        # 이미 존재하면 스킵
        if filepath.exists():
            print(f"[SKIP] {filename} - 이미 존재함")
            continue

        print(f"[생성중] {persona_id}...")

        try:
            # DALL-E 3로 이미지 생성
            response = client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size="1024x1024",
                quality="standard",
                n=1,
            )

            # 이미지 URL 가져오기
            image_url = response.data[0].url

            # 이미지 다운로드 및 저장
            with httpx.Client() as http_client:
                img_response = http_client.get(image_url)
                img_response.raise_for_status()

                with open(filepath, "wb") as f:
                    f.write(img_response.content)

            print(f"[완료] {filename} 저장됨")

        except Exception as e:
            print(f"[오류] {persona_id}: {e}")

    print("\n이미지 생성 완료!")


if __name__ == "__main__":
    # 프로젝트 루트로 이동
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    generate_and_save_images()
