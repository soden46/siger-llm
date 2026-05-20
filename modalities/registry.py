from __future__ import annotations

from .base import ModalitySpec


MODALITY_SPECS: dict[str, ModalitySpec] = {
    "text": ModalitySpec(
        name="text",
        family="language",
        input_unit="token_id",
        output_unit="token_id",
        adapter_kind="token_embedding",
        objective="causal_lm / seq2seq / masked_lm",
        status="implemented",
        notes="Current SigerLM path.",
    ),
    "pixel_lm": ModalitySpec(
        name="pixel_lm",
        family="vision",
        input_unit="image_patch_or_pixel_token",
        output_unit="image_patch_or_pixel_token",
        adapter_kind="patchify_or_vq_image_tokenizer",
        objective="next-patch prediction / masked patch prediction",
    ),
    "vision_encoder": ModalitySpec(
        name="vision_encoder",
        family="vision",
        input_unit="image_patch",
        output_unit="image_embedding",
        adapter_kind="patch_embed",
        objective="contrastive / classification / masked image modeling",
    ),
    "vision_language": ModalitySpec(
        name="vision_language",
        family="vision_language",
        input_unit="image_patch + text_token",
        output_unit="text_token",
        adapter_kind="vision_projector + text_embedding",
        objective="image captioning / visual QA / grounding",
    ),
    "image_generator": ModalitySpec(
        name="image_generator",
        family="generative_vision",
        input_unit="text_token_or_condition",
        output_unit="image_latent_or_patch_token",
        adapter_kind="latent_image_decoder",
        objective="diffusion / autoregressive image token generation",
    ),
    "speech_to_text": ModalitySpec(
        name="speech_to_text",
        family="speech",
        input_unit="audio_frame",
        output_unit="text_token",
        adapter_kind="audio_frontend + projector",
        objective="ctc / seq2seq transcription",
    ),
    "text_to_speech": ModalitySpec(
        name="text_to_speech",
        family="speech",
        input_unit="text_token",
        output_unit="mel_or_codec_token",
        adapter_kind="speech_codec_decoder",
        objective="codec token generation / mel regression",
    ),
    "audio_classification": ModalitySpec(
        name="audio_classification",
        family="audio",
        input_unit="audio_frame",
        output_unit="class_label",
        adapter_kind="audio_frontend + classifier",
        objective="cross_entropy",
    ),
    "speaker_recognition": ModalitySpec(
        name="speaker_recognition",
        family="audio",
        input_unit="audio_frame",
        output_unit="speaker_embedding_or_id",
        adapter_kind="audio_frontend + pooling_head",
        objective="contrastive / classification",
    ),
    "voice_emotion": ModalitySpec(
        name="voice_emotion",
        family="audio",
        input_unit="audio_frame",
        output_unit="emotion_label",
        adapter_kind="audio_frontend + classifier",
        objective="cross_entropy",
    ),
    "audio_captioning": ModalitySpec(
        name="audio_captioning",
        family="audio_language",
        input_unit="audio_frame",
        output_unit="text_token",
        adapter_kind="audio_frontend + text_decoder",
        objective="captioning",
    ),
    "voice_assistant": ModalitySpec(
        name="voice_assistant",
        family="speech_agent",
        input_unit="audio_frame + text_token",
        output_unit="text_token + speech_codec_token",
        adapter_kind="asr + dialogue + tts heads",
        objective="assistant instruction tuning",
    ),
    "video": ModalitySpec(
        name="video",
        family="video",
        input_unit="spatiotemporal_patch",
        output_unit="video_embedding_or_token",
        adapter_kind="tubelet_embed",
        objective="video understanding / masked video modeling",
    ),
    "audio_visual_omni": ModalitySpec(
        name="audio_visual_omni",
        family="omni",
        input_unit="text + image + audio + video sequence",
        output_unit="text + image/audio codec tokens",
        adapter_kind="multi-adapter fusion",
        objective="multimodal instruction tuning",
    ),
    "robotics_action": ModalitySpec(
        name="robotics_action",
        family="action",
        input_unit="observation_token + state_vector",
        output_unit="action_token_or_vector",
        adapter_kind="sensor_projector + action_head",
        objective="behavior cloning / sequence decision",
    ),
    "time_series_sensor": ModalitySpec(
        name="time_series_sensor",
        family="time_series",
        input_unit="sensor_timestep",
        output_unit="forecast_or_label",
        adapter_kind="numeric_projector",
        objective="forecasting / anomaly detection / classification",
    ),
    "code": ModalitySpec(
        name="code",
        family="language",
        input_unit="code_token",
        output_unit="code_token",
        adapter_kind="token_embedding",
        objective="causal_lm / instruction tuning",
        status="data_path_ready",
    ),
    "tabular": ModalitySpec(
        name="tabular",
        family="structured_data",
        input_unit="typed_cell_or_row_token",
        output_unit="label_or_cell_token",
        adapter_kind="schema_aware_projector",
        objective="classification / regression / table QA",
    ),
    "graph_kg": ModalitySpec(
        name="graph_kg",
        family="graph",
        input_unit="node_edge_path_token",
        output_unit="node_label_or_relation",
        adapter_kind="graph_encoder_projector",
        objective="link prediction / graph QA / retrieval",
    ),
    "retrieval_memory_agent": ModalitySpec(
        name="retrieval_memory_agent",
        family="agent",
        input_unit="query + memory_chunk + tool_event",
        output_unit="text_token + tool_call",
        adapter_kind="retriever + memory + tool_head",
        objective="RAG / tool-use instruction tuning",
        status="partially_implemented",
    ),
    "document_ocr": ModalitySpec(
        name="document_ocr",
        family="document_ai",
        input_unit="page_patch + layout_token",
        output_unit="text_token + bbox",
        adapter_kind="document_vision_projector",
        objective="OCR / document QA / layout extraction",
    ),
    "music_symbolic": ModalitySpec(
        name="music_symbolic",
        family="music",
        input_unit="midi_or_audio_codec_token",
        output_unit="midi_or_audio_codec_token",
        adapter_kind="music_tokenizer",
        objective="symbolic continuation / generation",
    ),
    "biological_sequence": ModalitySpec(
        name="biological_sequence",
        family="bio",
        input_unit="dna_rna_protein_token",
        output_unit="sequence_token_or_label",
        adapter_kind="bio_tokenizer",
        objective="sequence modeling / function prediction",
    ),
    "financial_event_sequence": ModalitySpec(
        name="financial_event_sequence",
        family="event_sequence",
        input_unit="event_or_numeric_timestep",
        output_unit="forecast_or_event_token",
        adapter_kind="event_projector",
        objective="forecasting / event prediction / risk classification",
    ),
}


def get_modality_spec(name: str) -> ModalitySpec:
    key = name.lower()
    if key not in MODALITY_SPECS:
        raise KeyError(f"Unknown modality: {name}")
    return MODALITY_SPECS[key]


def list_modalities() -> list[str]:
    return sorted(MODALITY_SPECS)
