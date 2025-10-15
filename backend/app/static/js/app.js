(function () {
  const TARGET_SAMPLE_RATE = 16000;
  const chatLog = document.getElementById('chat-log');
  const chatForm = document.getElementById('chat-form');
  const chatInput = document.getElementById('chat-input');
  const chatSend = document.getElementById('chat-send');
  const chatStatus = document.getElementById('chat-status');
  const recordBtn = document.getElementById('chat-record');
  let history = [];
  let audioContext;
  let mediaRecorder;
  let recordingStream;
  let audioChunks = [];
  let isRecording = false;
  let pendingUserEntry = null;
  const teacherAudioUrls = new Set();

  function cleanupTeacherAudioUrls() {
    teacherAudioUrls.forEach((url) => URL.revokeObjectURL(url));
    teacherAudioUrls.clear();
  }

  window.addEventListener('beforeunload', cleanupTeacherAudioUrls);

  function ensureAudioContext() {
    if (!audioContext) {
      const Ctx = window.AudioContext || window.webkitAudioContext;
      if (Ctx) {
        audioContext = new Ctx();
      }
    }
    return audioContext;
  }

  function setStatus(message) {
    chatStatus.textContent = message || '';
  }

  function appendMessage(role, content, options = {}) {
    const entry = document.createElement('div');
    entry.className = role === 'assistant' ? 'msg msg-assistant' : 'msg msg-user';
    entry.dataset.role = role;
    const label = document.createElement('strong');
    label.textContent = role === 'assistant' ? 'Tutor:' : 'You:';
    entry.appendChild(label);
    const span = document.createElement('span');
    span.className = 'msg-text';
    span.textContent = ` ${content}`;
    entry.appendChild(span);
    if (options.focusPhrase) {
      appendFocus(entry, options.focusPhrase, options.focusTranslation);
    }
    if (options.spokenText || options.spokenAudioUrl) {
      appendSpoken(entry, options.spokenText, options.spokenAudioUrl);
    }
    chatLog.appendChild(entry);
    chatLog.scrollTop = chatLog.scrollHeight;
    return entry;
  }

  function appendFocus(entry, phrase, translation) {
    if (!phrase) return;
    const focus = document.createElement('div');
    focus.className = 'msg-focus';
    focus.textContent = 'Focus phrase: ';
    const laoSpan = document.createElement('span');
    laoSpan.className = 'lao';
    laoSpan.textContent = phrase;
    focus.appendChild(laoSpan);
    if (translation) {
      const translationSpan = document.createElement('span');
      translationSpan.textContent = ` · ${translation}`;
      focus.appendChild(translationSpan);
    }
    entry.appendChild(focus);
  }

  function appendSpoken(entry, spokenText, audioUrl) {
    if (!spokenText && !audioUrl) return;
    const spoken = document.createElement('div');
    spoken.className = 'msg-spoken';
    const icon = document.createElement('span');
    icon.setAttribute('aria-hidden', 'true');
    icon.textContent = '🎧';
    spoken.appendChild(icon);
    const text = document.createElement('span');
    text.textContent = spokenText ? `Spoken reply: ${spokenText}` : 'Spoken reply available';
    spoken.appendChild(text);
    if (audioUrl) {
      const audioEl = document.createElement('audio');
      audioEl.controls = true;
      audioEl.preload = 'auto';
      audioEl.src = audioUrl;
      audioEl.setAttribute('aria-label', 'Tutor pronunciation playback');
      spoken.appendChild(audioEl);
      setTimeout(() => {
        audioEl.play().catch((error) => {
          console.warn('Automatic tutor playback blocked', error);
        });
      }, 0);
    }
    entry.appendChild(spoken);
  }

  function appendUtteranceFeedback(entry, feedback) {
    if (!feedback) return;
    const wrap = document.createElement('div');
    wrap.className = 'msg-feedback';
    const details = [];
    if (feedback.lao_text) {
      details.push(`Heard: ${feedback.lao_text}`);
    }
    if (feedback.romanised) {
      details.push(`Romanisation: ${feedback.romanised}`);
    }
    if (feedback.translation) {
      details.push(`Meaning: ${feedback.translation}`);
    }
    wrap.textContent = details.join(' · ');
    if (feedback.corrections && feedback.corrections.length) {
      const list = document.createElement('ul');
      feedback.corrections.forEach((hint) => {
        const item = document.createElement('li');
        item.textContent = hint;
        list.appendChild(item);
      });
      wrap.appendChild(list);
    }
    if (feedback.praise) {
      const praise = document.createElement('div');
      praise.textContent = feedback.praise;
      wrap.appendChild(praise);
    }
    entry.appendChild(wrap);
  }

  function updateMessage(entry, content) {
    if (!entry) return;
    const span = entry.querySelector('.msg-text');
    if (span) {
      span.textContent = ` ${content}`;
    }
  }

  function base64ToFloat32(base64) {
    const binary = atob(base64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i += 1) {
      bytes[i] = binary.charCodeAt(i);
    }
    if (bytes.length % 4 !== 0) {
      throw new Error('Teacher audio payload is malformed');
    }
    return new Float32Array(bytes.buffer);
  }

  function float32ToWavBytes(floatArray, sampleRate) {
    const bytesPerSample = 2;
    const blockAlign = bytesPerSample;
    const dataSize = floatArray.length * bytesPerSample;
    const buffer = new ArrayBuffer(44 + dataSize);
    const view = new DataView(buffer);
    let offset = 0;

    const writeString = (str) => {
      for (let i = 0; i < str.length; i += 1) {
        view.setUint8(offset + i, str.charCodeAt(i));
      }
      offset += str.length;
    };

    writeString('RIFF');
    view.setUint32(offset, 36 + dataSize, true);
    offset += 4;
    writeString('WAVE');
    writeString('fmt ');
    view.setUint32(offset, 16, true);
    offset += 4;
    view.setUint16(offset, 1, true);
    offset += 2;
    view.setUint16(offset, 1, true);
    offset += 2;
    view.setUint32(offset, sampleRate, true);
    offset += 4;
    view.setUint32(offset, sampleRate * blockAlign, true);
    offset += 4;
    view.setUint16(offset, blockAlign, true);
    offset += 2;
    view.setUint16(offset, bytesPerSample * 8, true);
    offset += 2;
    writeString('data');
    view.setUint32(offset, dataSize, true);
    offset += 4;

    const pcm = new Int16Array(buffer, offset, floatArray.length);
    for (let i = 0; i < floatArray.length; i += 1) {
      const sample = Math.max(-1, Math.min(1, floatArray[i]));
      pcm[i] = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
    }

    return new Uint8Array(buffer);
  }

  function prepareTeacherAudioClip(base64, sampleRate) {
    const floatData = base64ToFloat32(base64);
    if (!floatData.length) {
      throw new Error('Empty teacher audio payload');
    }
    const sr = Number(sampleRate) || TARGET_SAMPLE_RATE;
    const wavBytes = float32ToWavBytes(floatData, sr);
    const blob = new Blob([wavBytes], { type: 'audio/wav' });
    const url = URL.createObjectURL(blob);
    teacherAudioUrls.add(url);
    return { url, duration: floatData.length / sr };
  }

  function floatToPcm16Base64(floatArray) {
    const buffer = new ArrayBuffer(floatArray.length * 2);
    const view = new DataView(buffer);
    for (let i = 0; i < floatArray.length; i += 1) {
      const sample = Math.max(-1, Math.min(1, floatArray[i]));
      view.setInt16(i * 2, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
    }
    const bytes = new Uint8Array(buffer);
    let binary = '';
    const chunkSize = 0x8000;
    for (let i = 0; i < bytes.length; i += chunkSize) {
      const chunk = bytes.subarray(i, i + chunkSize);
      binary += String.fromCharCode.apply(null, Array.from(chunk));
    }
    return btoa(binary);
  }

  function downsampleToTarget(audioBuffer) {
    const sourceRate = audioBuffer.sampleRate;
    const channelData = audioBuffer.getChannelData(0);
    if (sourceRate === TARGET_SAMPLE_RATE) {
      return new Float32Array(channelData);
    }
    const ratio = sourceRate / TARGET_SAMPLE_RATE;
    const length = Math.round(channelData.length / ratio);
    const result = new Float32Array(length);
    let offsetResult = 0;
    let offsetBuffer = 0;
    while (offsetResult < length) {
      const nextOffsetBuffer = Math.min(
        channelData.length,
        Math.round((offsetResult + 1) * ratio),
      );
      let accum = 0;
      let count = 0;
      for (let i = offsetBuffer; i < nextOffsetBuffer; i += 1) {
        accum += channelData[i];
        count += 1;
      }
      result[offsetResult] = count > 0 ? accum / count : 0;
      offsetResult += 1;
      offsetBuffer = nextOffsetBuffer;
    }
    return result;
  }

  async function sendConversation(payload, { userEntry } = {}) {
    chatSend.disabled = true;
    if (recordBtn) {
      recordBtn.disabled = true;
    }
    setStatus('Thinking…');
    try {
      const response = await fetch('/api/v1/conversation', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...payload, history }),
      });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const data = await response.json();
      history = data.history || [];
      if (userEntry && data.heard_text) {
        updateMessage(userEntry, `🎤 ${data.heard_text}`);
      }
      let spokenAudioUrl = null;
      if (data.teacher_audio_base64 && data.teacher_audio_sample_rate) {
        try {
          const clip = prepareTeacherAudioClip(
            data.teacher_audio_base64,
            data.teacher_audio_sample_rate,
          );
          spokenAudioUrl = clip.url;
        } catch (err) {
          console.warn('Failed to prepare tutor audio', err);
        }
      }
      const replyEntry = appendMessage(
        'assistant',
        data.reply?.content || 'I am still getting ready to chat.',
        {
          focusPhrase: data.focus_phrase,
          focusTranslation: data.focus_translation,
          spokenText: data.spoken_text,
          spokenAudioUrl,
        },
      );
      appendUtteranceFeedback(replyEntry, data.utterance_feedback);
    } catch (error) {
      console.error(error);
      appendMessage(
        'assistant',
        'I ran into a problem understanding that. Please try again after a moment.',
      );
    } finally {
      chatSend.disabled = false;
      if (recordBtn) {
        recordBtn.disabled = false;
      }
      setStatus('');
    }
  }

  chatForm.addEventListener('submit', async (event) => {
    event.preventDefault();
    const message = chatInput.value.trim();
    if (!message) return;
    const userEntry = appendMessage('user', message);
    chatInput.value = '';
    chatInput.focus();
    await sendConversation({ message }, { userEntry });
  });

  async function startRecording() {
    if (!navigator.mediaDevices || !window.MediaRecorder) {
      appendMessage(
        'assistant',
        'Microphone recording is not supported in this browser.',
      );
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      recordingStream = stream;
      const options = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
        ? { mimeType: 'audio/webm;codecs=opus' }
        : undefined;
      mediaRecorder = new MediaRecorder(stream, options);
      audioChunks = [];
      mediaRecorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          audioChunks.push(event.data);
        }
      };
      mediaRecorder.onstart = () => {
        pendingUserEntry = appendMessage('user', '🎤 Listening…');
        setStatus('Recording… tap stop when finished.');
        chatSend.disabled = true;
      };
      mediaRecorder.onstop = async () => {
        setStatus('Processing audio…');
        chatSend.disabled = false;
        if (recordBtn) {
          recordBtn.disabled = true;
        }
        try {
          if (pendingUserEntry) {
            updateMessage(pendingUserEntry, '🎤 Processing audio…');
          }
          const blob = new Blob(audioChunks, { type: mediaRecorder.mimeType });
          const arrayBuffer = await blob.arrayBuffer();
          const ctx = ensureAudioContext();
          if (!ctx) {
            appendMessage(
              'assistant',
              'Audio playback is not supported in this environment.',
            );
            return;
          }
          const audioBuffer = await ctx.decodeAudioData(arrayBuffer.slice(0));
          const floatData = downsampleToTarget(audioBuffer);
          const base64 = floatToPcm16Base64(floatData);
          await sendConversation(
            { audio_base64: base64, sample_rate: TARGET_SAMPLE_RATE },
            { userEntry: pendingUserEntry },
          );
        } catch (error) {
          console.error(error);
          appendMessage(
            'assistant',
            'I could not process that audio clip. Please try again.',
          );
        } finally {
          if (recordBtn) {
            recordBtn.disabled = false;
            recordBtn.textContent = '🎙️ Record phrase';
          }
          setStatus('');
          pendingUserEntry = null;
          if (recordingStream) {
            recordingStream.getTracks().forEach((track) => track.stop());
            recordingStream = null;
          }
        }
      };
      mediaRecorder.start();
      isRecording = true;
      if (recordBtn) {
        recordBtn.textContent = '⏹️ Stop recording';
      }
    } catch (error) {
      console.error(error);
      appendMessage('assistant', 'Microphone permission was denied or not available.');
    }
  }

  function stopRecording() {
    if (mediaRecorder && isRecording) {
      mediaRecorder.stop();
      isRecording = false;
    }
  }

  if (recordBtn) {
    if (!navigator.mediaDevices || !window.MediaRecorder) {
      recordBtn.disabled = true;
      recordBtn.textContent = 'Recording unavailable';
    }
    recordBtn.addEventListener('click', async () => {
      if (isRecording) {
        stopRecording();
      } else {
        await startRecording();
      }
    });
  }
})();
