import os
from pathlib import Path
import subprocess
from faster_whisper import WhisperModel
import logging
from tqdm import tqdm
import time
import re
import torch
import concurrent.futures
from typing import List, Tuple
import tempfile
import shutil

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

class SubtitleGenerator:
    def __init__(self, model_size: str = "base", force: bool = False, language: str = None,
                 device: str = "auto", compute_type: str = "auto", threads: int = 0,
                 parallel_segments: int = 4):
        """
        Инициализация генератора субтитров
        
        Args:
            model_size: Размер модели Whisper ("tiny", "base", "small", "medium", "large")
            force: Принудительная перезапись существующих файлов
            language: Язык аудио (например, "ru", "en"). Если None, будет определен автоматически
            device: Устройство для вычислений ("cpu", "cuda", "auto")
            compute_type: Тип вычислений ("int8", "float16", "float32", "auto")
            threads: Количество потоков для CPU (0 = автоматически)
            parallel_segments: Количество параллельно обрабатываемых сегментов
        """
        self.force = force
        self.language = language
        self.parallel_segments = parallel_segments
        
        # Определяем устройство и тип вычислений
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
        if compute_type == "auto":
            if device == "cuda":
                compute_type = "float16"  # Оптимально для GPU
            else:
                compute_type = "int8"  # Оптимально для CPU
                
        logger.info(f"Используется устройство: {device}")
        if device == "cuda":
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"Вычислительный тип: {compute_type}")
        else:
            logger.info(f"Количество потоков CPU: {threads if threads > 0 else 'автоматически'}")
            
        logger.info(f"Загрузка модели Whisper {model_size}...")
        start_time = time.time()
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            cpu_threads=threads
        )
        load_time = time.time() - start_time
        logger.info(f"Модель загружена успешно за {load_time:.1f} секунд")

    def check_audio_stream(self, video_path: str) -> bool:
        """
        Проверка наличия аудио потока в видео
        
        Args:
            video_path: Путь к видео файлу
            
        Returns:
            bool: True если аудио поток найден
        """
        cmd = [
            "ffprobe", "-v", "error", "-select_streams", "a",
            "-show_entries", "stream=codec_type", "-of", "csv=p=0",
            video_path
        ]
        try:
            output = subprocess.check_output(cmd).decode().strip()
            has_audio = bool(output)
            if not has_audio:
                logger.error("В видео не найден аудио поток!")
            return has_audio
        except subprocess.CalledProcessError:
            logger.error("Ошибка при проверке аудио потока")
            return False

    def extract_audio(self, video_path: str, audio_path: str) -> str:
        """
        Извлечение аудио из видео
        
        Args:
            video_path: Путь к видео файлу
            audio_path: Путь для сохранения аудио
            
        Returns:
            str: Путь к аудио файлу
        """
        # Проверяем существование аудио файла
        if os.path.exists(audio_path) and not self.force:
            logger.info(f"Аудио файл {audio_path} уже существует, пропускаем извлечение")
            return audio_path
            
        logger.info("Извлечение аудио из видео...")
        start_time = time.time()
        
        # Проверяем наличие аудио в видео
        check_audio_cmd = [
            "ffprobe", "-v", "error", "-select_streams", "a:0",
            "-show_entries", "stream=codec_type", "-of", "default=noprint_wrappers=1:nokey=1",
            video_path
        ]
        try:
            has_audio = subprocess.check_output(check_audio_cmd).decode().strip() == "audio"
        except subprocess.CalledProcessError:
            has_audio = False
            
        if not has_audio:
            raise ValueError(f"В видео файле {video_path} не обнаружено аудио")
            
        # Извлекаем аудио с максимальным качеством
        process = subprocess.Popen([
            "ffmpeg", "-i", video_path,
            "-vn",  # Без видео
            "-acodec", "libmp3lame",  # Кодек MP3
            "-q:a", "0",  # Максимальное качество
            "-af", "highpass=f=200,lowpass=f=3000",  # Фильтры для улучшения голоса
            "-ac", "2",  # Стерео
            "-y",  # Перезаписать, если файл существует
            audio_path
        ], stderr=subprocess.PIPE)
        
        # Создаем прогресс-бар
        pbar = tqdm(total=100, desc="Извлечение аудио", unit="%")
        
        # Мониторим прогресс
        while True:
            output = process.stderr.readline().decode()
            if output == "" and process.poll() is not None:
                break
            if "time=" in output:
                try:
                    time_str = re.search(r"time=(\d+:\d+:\d+\.\d+)", output).group(1)
                    h, m, s = map(float, time_str.split(":"))
                    total_seconds = h * 3600 + m * 60 + s
                    progress = (total_seconds / self._get_video_duration(video_path)) * 100
                    pbar.update(int(progress) - pbar.n)
                except:
                    pass
        
        pbar.close()
        process.wait()
        
        if process.returncode != 0:
            raise RuntimeError(f"Ошибка при извлечении аудио: {process.stderr.read().decode()}")
            
        extract_time = time.time() - start_time
        logger.info(f"Аудио сохранено в {audio_path} за {extract_time:.1f} секунд")
        return audio_path

    def split_audio(self, audio_path: str, segment_duration: int = 300) -> List[str]:
        """
        Разделение аудио на сегменты
        
        Args:
            audio_path: Путь к аудио файлу
            segment_duration: Длительность сегмента в секундах
            
        Returns:
            List[str]: Список путей к сегментам
        """
        # Создаем временную директорию для сегментов
        temp_dir = tempfile.mkdtemp()
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        
        # Получаем длительность аудио
        duration_cmd = [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", audio_path
        ]
        duration = float(subprocess.check_output(duration_cmd).decode().strip())
        
        # Вычисляем количество сегментов
        num_segments = 2  # Фиксированное количество сегментов
        segment_duration = duration / num_segments
        logger.info(f"Разделение аудио на {num_segments} сегмента по {segment_duration:.1f} секунд")
        
        # Создаем прогресс-бар
        pbar = tqdm(total=num_segments, desc="Разделение аудио", unit="сегмент")
        
        # Разделяем аудио на сегменты
        segments = []
        for i in range(num_segments):
            start_time = i * segment_duration
            segment_path = os.path.join(temp_dir, f"{base_name}_segment_{i:03d}.mp3")
            
            process = subprocess.Popen([
                "ffmpeg", "-i", audio_path,
                "-ss", str(start_time),
                "-t", str(segment_duration),
                "-acodec", "libmp3lame",
                "-q:a", "0",
                "-y",
                segment_path
            ], stderr=subprocess.PIPE)
            process.wait()
            
            if process.returncode == 0:
                segments.append(segment_path)
                logger.info(f"Создан сегмент {i+1}/{num_segments}: {os.path.basename(segment_path)}")
            pbar.update(1)
        
        pbar.close()
        logger.info(f"Всего создано {len(segments)} сегментов")
        return segments

    def process_segment(self, segment_path: str, pbar: tqdm = None) -> List[Tuple[float, float, str]]:
        """
        Обработка одного сегмента аудио
        
        Args:
            segment_path: Путь к сегменту аудио
            pbar: Прогресс-бар для этого сегмента
            
        Returns:
            List[Tuple[float, float, str]]: Список (начало, конец, текст) для каждого сегмента
        """
        logger.info(f"Начало обработки сегмента {os.path.basename(segment_path)}")
        segments, info = self.model.transcribe(
            segment_path,
            language=self.language,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            beam_size=5,
            condition_on_previous_text=True,
            temperature=0.0,
            compression_ratio_threshold=2.4,
            no_speech_threshold=0.6
        )
        
        # Получаем базовое время сегмента
        base_time = float(os.path.basename(segment_path).split("_")[-1].split(".")[0]) * (300 / 2)  # Исправляем базовое время
        
        # Собираем результаты
        results = []
        total_duration = 0
        for segment in segments:
            duration = segment.end - segment.start
            total_duration += duration
            results.append((
                base_time + segment.start,
                base_time + segment.end,
                segment.text.strip()
            ))
            if pbar:
                pbar.update(1)
            
        # Логируем информацию о сегменте
        logger.info(f"Сегмент {os.path.basename(segment_path)}: обработано {len(results)} фрагментов, общая длительность речи: {total_duration:.1f}с")
        
        return results

    def generate_subtitles(self, audio_path: str, output_path: str = None) -> str:
        """
        Генерация субтитров из аудио
        
        Args:
            audio_path: Путь к аудио файлу
            output_path: Путь для сохранения субтитров (опционально)
            
        Returns:
            str: Путь к файлу субтитров
        """
        if output_path is None:
            output_path = str(Path(audio_path).with_suffix(".srt"))
            
        # Проверяем существование файла
        if os.path.exists(output_path) and not self.force:
            logger.info(f"Файл субтитров {output_path} уже существует, пропускаем генерацию")
            return output_path
            
        logger.info("Генерация субтитров...")
        start_time = time.time()
        
        # Получаем длительность аудио
        duration_cmd = [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", audio_path
        ]
        duration = float(subprocess.check_output(duration_cmd).decode().strip())
        logger.info(f"Обработка аудио длительностью {self._format_timestamp(duration)}")
        
        # Создаем прогресс-бар
        pbar = tqdm(total=100, desc="Генерация субтитров", unit="%")
        
        # Используем указанный язык или определяем автоматически
        logger.info("Начало распознавания речи...")
        segments, info = self.model.transcribe(
            audio_path,
            language=self.language,
            vad_filter=True,  # Фильтр голосовой активности
            vad_parameters=dict(min_silence_duration_ms=500),  # Минимальная длительность тишины
            beam_size=5,  # Увеличиваем размер луча для лучшего качества
            condition_on_previous_text=True,  # Учитываем предыдущий текст
            temperature=0.0,  # Отключаем случайность для стабильности
            compression_ratio_threshold=2.4,  # Порог сжатия для лучшего качества
            no_speech_threshold=0.6  # Порог определения речи
        )
        
        if self.language is None:
            logger.info(f"Определен язык: {info.language}")
        
        # Собираем результаты
        all_results = []
        last_end = 0
        total_silence = 0
        for segment in segments:
            # Логируем информацию о сегменте
            segment_duration = segment.end - segment.start
            silence_before = segment.start - last_end if last_end > 0 else 0
            total_silence += silence_before
            
            logger.info(f"Фрагмент: {self._format_timestamp(segment.start)} - {self._format_timestamp(segment.end)} "
                       f"(длительность: {segment_duration:.1f}с, тишина до: {silence_before:.1f}с)")
            logger.info(f"Текст: {segment.text.strip()}")
            
            all_results.append((
                segment.start,
                segment.end,
                segment.text.strip()
            ))
            
            # Обновляем прогресс
            progress = (segment.end / duration) * 100
            pbar.update(int(progress) - pbar.n)
            last_end = segment.end
        
        pbar.close()
        
        # Логируем итоговую статистику
        total_speech = sum(end - start for start, end, _ in all_results)
        logger.info(f"Статистика распознавания:")
        logger.info(f"- Всего фрагментов: {len(all_results)}")
        logger.info(f"- Общая длительность речи: {total_speech:.1f}с")
        logger.info(f"- Общая длительность тишины: {total_silence:.1f}с")
        logger.info(f"- Процент речи: {(total_speech / duration * 100):.1f}%")
        
        # Сортируем результаты по времени
        all_results.sort(key=lambda x: x[0])
        
        # Записываем субтитры
        with open(output_path, "w", encoding="utf-8") as f:
            for i, (start, end, text) in enumerate(all_results, 1):
                f.write(f"{i}\n")
                f.write(f"{self._format_timestamp(start)} --> {self._format_timestamp(end)}\n")
                f.write(f"{text}\n\n")
        
        generate_time = time.time() - start_time
        logger.info(f"Субтитры сохранены в {output_path} за {generate_time:.1f} секунд")
        return output_path

    def _format_timestamp(self, seconds: float) -> str:
        """
        Форматирование времени в формат SRT (HH:MM:SS,mmm)
        
        Args:
            seconds: Время в секундах
            
        Returns:
            str: Отформатированное время
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        milliseconds = int((seconds % 1) * 1000)
        seconds = int(seconds)
        
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Генерация субтитров из видео с помощью Whisper")
    parser.add_argument("video_path", help="Путь к видео файлу")
    parser.add_argument("--model-size", default="medium", choices=["tiny", "base", "small", "medium", "large"],
                      help="Размер модели Whisper (по умолчанию: medium)")
    parser.add_argument("--output-dir", help="Директория для сохранения результатов (опционально)")
    parser.add_argument("--force", action="store_true", help="Принудительная перезапись существующих файлов")
    parser.add_argument("--language", help="Язык аудио (например, 'ru', 'en'). Если не указан, будет определен автоматически")
    parser.add_argument("--device", default="auto", choices=["cpu", "cuda", "auto"],
                      help="Устройство для вычислений (по умолчанию: auto)")
    parser.add_argument("--compute-type", default="auto", choices=["int8", "float16", "float32", "auto"],
                      help="Тип вычислений (по умолчанию: auto)")
    parser.add_argument("--threads", type=int, default=0,
                      help="Количество потоков для CPU (0 = автоматически)")
    parser.add_argument("--parallel-segments", type=int, default=4,
                      help="Количество параллельно обрабатываемых сегментов (по умолчанию: 4)")
    
    args = parser.parse_args()
    
    # Создаем директорию для результатов, если указана
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        output_dir = args.output_dir
    else:
        output_dir = os.path.dirname(args.video_path)
    
    # Инициализируем генератор
    generator = SubtitleGenerator(
        model_size=args.model_size,
        force=args.force,
        language=args.language,
        device=args.device,
        compute_type=args.compute_type,
        threads=args.threads,
        parallel_segments=args.parallel_segments
    )
    
    # Извлекаем аудио
    audio_path = os.path.join(output_dir, os.path.splitext(os.path.basename(args.video_path))[0] + ".mp3")
    if not os.path.exists(audio_path) or args.force:
        generator.extract_audio(args.video_path, audio_path)
    else:
        logger.info(f"Используется существующий аудио файл: {audio_path}")
    
    # Генерируем субтитры
    subtitle_path = os.path.join(output_dir, os.path.splitext(os.path.basename(args.video_path))[0] + ".srt")
    generator.generate_subtitles(audio_path, subtitle_path)
    
    # Удаляем временный аудио файл, только если он был создан в этом запуске
    if os.path.exists(audio_path) and args.force:
        os.remove(audio_path)
        logger.info(f"Временный аудио файл {audio_path} удален")
    
    logger.info("Готово! Субтитры успешно сгенерированы.")

if __name__ == "__main__":
    main()
