import io
import os
import streamlit as st
from openai import OpenAI
from streamlit_mic_recorder import mic_recorder

# 1. APIキーの設定（環境変数 OPENAI_API_KEY を使用、またはここに直接記述）
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY"))

# 2. ページ初期設定
st.set_page_config(page_title="部屋干し洗剤 AIインタビュー", layout="centered")
st.title("🧺 部屋干し洗剤 AIインタビュー")

# 3. システムプロンプトの定義
SYSTEM_PROMPT = """
あなたは家庭の洗濯・部屋干しの実態を調査する、親しみやすく聞き上手な「AIインタビュアー」です。
現在、部屋干し作業を行っているユーザーに対し、音声対話を通じて「部屋干し洗剤に対するリアルな不満や地味なストレス」を3回のやり取り（ターン）で聞き出してください。

【ルール】
- 1回の発話は相づち・共感 ＋ 質問1つ（80文字以内）。
- Turn 1: 洗剤名と最初の不満を聞く。
- Turn 2: 不満の具体的シーン（干した直後、乾いた後、ボトル容器など）を深掘りする。
- Turn 3: 理想の機能を聞き、「以上でアンケートは終了です！」と告げて締めくくる。
- 3回目の応答が終わったら、必ず「以上でアンケートは終了です！」を含めてください。
"""

# 4. セッション状態（会話履歴・ターン数）の初期化
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    # 初期挨拶（Turn 1の呼びかけ）
    initial_greeting = "お疲れ様です！部屋干し中ですか？今お使いの洗剤名と、部屋干ししてて『ここがちょっと嫌だな』って思う瞬間があれば教えてください！"
    st.session_state.messages.append({"role": "assistant", "content": initial_greeting})
    st.session_state.turn_count = 1
    st.session_state.is_finished = False

# 5. 会話ログの表示
st.write("---")
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    elif msg["role"] == "assistant":
        st.chat_message("assistant").write(msg["content"])

st.write("---")

# 6. インタビュー終了判定
if st.session_state.is_finished:
    st.success("🎉 アンケートは終了しました。ご協力ありがとうございました！")
    st.info("【特典ギフトコード】 WA6S-8823-9012 （Amazonギフトカード 300円分）")
    st.stop()

# 7. 音声録音 UI（スマホマイク対応）
st.subheader("🎤 音声で回答する")
st.caption("ボタンを押して話し、話し終わったらもう一度押して送信してください。")

audio = mic_recorder(
    start_prompt="🎤 録音開始",
    stop_prompt="⏹️ 録音完了・送信",
    key="recorder"
)

# 8. 音声データが送信された場合の処理
if audio:
    # 録音データをバイト形式で取得
    audio_bytes = audio["bytes"]
    audio_bio = io.BytesIO(audio_bytes)
    audio_bio.name = "input.wav"

    with st.spinner("AIが声を解析中..."):
        # A. Whisperによる文字起こし (STT)
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_bio,
            language="ja"
        )
        user_text = transcript.text

        # ユーザーの発話を画面・履歴に追加
        st.session_state.messages.append({"role": "user", "content": user_text})

        # B. GPT-4oによるテキスト応答生成 (LLM)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=st.session_state.messages,
            max_tokens=150,
            temperature=0.7
        )
        ai_text = response.choices[0].message.content
        st.session_state.messages.append({"role": "assistant", "content": ai_text})
        st.session_state.turn_count += 1

        # C. OpenAI TTSによる音声合成 (TTS)
        tts_response = client.audio.speech.create(
            model="tts-1",
            voice="nova",  # 親しみやすい女性声（nova, alloy, shimmer等）
            input=ai_text
        )
        audio_out_bytes = tts_response.content

        # D. 音声の自動再生と画面更新
        st.audio(audio_out_bytes, format="audio/mp3", autoplay=True)

        # 終了フラグの判定
        if "以上でアンケートは終了です" in ai_text or st.session_state.turn_count > 3:
            st.session_state.is_finished = True

        st.rerun()