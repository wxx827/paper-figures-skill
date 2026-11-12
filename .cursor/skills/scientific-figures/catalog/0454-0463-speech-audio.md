# 语音音频 · Speech & audio  `454-463`

10 figures. Run any of them from `scripts/`; the PNG lands in `output/` under the same number.

| # | Script | What it draws | Use when |
|---|--------|---------------|----------|
| 454 | `fig_spx_waveform_spectrogram.py` | 454 - Waveform with dual-resolution spectrograms of a synthetic utterance | 语音音频 |
| 455 | `fig_spx_mel_mfcc.py` | 455 - The mel-frequency feature chain, one transform per panel | 语音音频 |
| 456 | `fig_spx_phoneme_alignment.py` | 456 - Forced alignment: spectrogram over a phone tier over a word tier | 语音音频 |
| 457 | `fig_spx_speaker_embedding.py` | 457 - Speaker embedding space: t-SNE map, centroid similarity, EER | 语音音频 |
| 458 | `fig_spx_denoise_compare.py` | 458 - Speech enhancement compared row by row, on one shared colour scale | 语音音频 |
| 459 | `fig_spx_source_separation.py` | 459 - Three-speaker separation: mixture, estimates, ideal masks, BSS scores | 语音音频 |
| 460 | `fig_spx_prosody_contour.py` | 460 - Prosody of one sentence: F0, energy and syllable rate on one clock | 语音音频 |
| 461 | `fig_spx_asr_errors.py` | 461 - ASR error analysis: word alignment, error mix, stratified WER | 语音音频 |
| 462 | `fig_spx_music_structure.py` | Chroma self-similarity matrix with time-aligned section ribbon, checkerboard novelty boundaries and chromagram | 音乐信息检索, 段落切分, "音乐结构" |
| 463 | `fig_spx_vocoder_compare.py` | Five vocoder outputs on one shared dB scale over their error maps on a second shared symmetric scale, with MOS, RTF and measured LSD per column | 语音合成对比, 语谱误差, "声码器对比" |
