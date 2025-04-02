#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
from pathlib import Path
from deep_translator import GoogleTranslator
from tqdm import tqdm
import re

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

class SubtitleTranslator:
    def __init__(self, src_lang='auto', dest_lang='ru'):
        """
        Инициализация переводчика субтитров
        
        Args:
            src_lang (str): исходный язык (по умолчанию 'auto' - автоматическое определение)
            dest_lang (str): язык перевода (по умолчанию 'ru' - русский)
        """
        self.translator = GoogleTranslator(source=src_lang, target=dest_lang)
        self.src_lang = src_lang
        self.dest_lang = dest_lang
        self.pattern = re.compile(r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n(.*?)(?=\n\n|\Z)', re.DOTALL)

    def translate_text(self, text):
        """
        Перевод текста
        
        Args:
            text (str): текст для перевода
            
        Returns:
            str: переведенный текст
        """
        try:
            translation = self.translator.translate(text)
            return translation
        except Exception as e:
            logger.error(f"Ошибка при переводе текста: {e}")
            return text

    def translate_subtitles(self, input_file, output_file=None):
        """
        Перевод субтитров из файла
        
        Args:
            input_file (str): путь к входному файлу субтитров
            output_file (str, optional): путь к выходному файлу. Если не указан, будет создан автоматически
            
        Returns:
            str: путь к переведенному файлу
        """
        input_path = Path(input_file)
        
        if not output_file:
            # Создаем имя выходного файла, добавляя суффикс с языком перевода
            output_file = input_path.parent / f"{input_path.stem}_{self.dest_lang}{input_path.suffix}"
        
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Разбиваем на блоки субтитров
            blocks = self.pattern.findall(content)
            
            logger.info(f"Найдено {len(blocks)} блоков субтитров для перевода")
            
            translated_blocks = []
            for block in tqdm(blocks, desc="Перевод субтитров"):
                num, start, end, text = block
                # Удаляем лишние пробелы и переносы строк
                text = text.strip()
                # Переводим текст
                translated_text = self.translate_text(text)
                # Формируем блок с переведенным текстом
                translated_block = f"{num}\n{start} --> {end}\n{translated_text}\n\n"
                translated_blocks.append(translated_block)
            
            # Записываем результат
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(''.join(translated_blocks))
            
            logger.info(f"Перевод завершен. Результат сохранен в {output_file}")
            return str(output_file)
            
        except Exception as e:
            logger.error(f"Ошибка при обработке файла {input_file}: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description='Переводчик субтитров')
    parser.add_argument('input_file', help='Путь к файлу субтитров')
    parser.add_argument('--output', '-o', help='Путь к выходному файлу (опционально)')
    parser.add_argument('--src-lang', default='auto', help='Исходный язык (по умолчанию: auto)')
    parser.add_argument('--dest-lang', default='ru', help='Язык перевода (по умолчанию: ru)')
    
    args = parser.parse_args()
    
    translator = SubtitleTranslator(src_lang=args.src_lang, dest_lang=args.dest_lang)
    translator.translate_subtitles(args.input_file, args.output)

if __name__ == '__main__':
    main() 