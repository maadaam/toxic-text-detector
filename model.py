import joblib
import pymorphy3
import re

class ToxityModel:
    def __init__(self, russian_stop_words_path: "data/russian_stop_words", model_path: "artefacts/toxic_model_v1.pkl"):
        self.russian_stop_words = joblib.load(russian_stop_words_path) #загружаем стоп слова
        self.pipeline = joblib.load(model_path) #загружаем модель для предсказаний (+ векторизатор)
        self.morph = pymorphy3.MorphAnalyzer() #загружаем лемматизатор


    def preprocess(self, text: str) -> str: #функция предобработки текста
        if not isinstance(text, str):
            return ""
        text = text.lower()

        # Удаление символов, кроме букв и пробелов
        text = re.sub(r'[^а-яё\s]', ' ', text)

        # Удаление лишних пробелов
        text = re.sub(r'\s+', ' ', text).strip()

        # Токенизация
        tokens = text.split()

        # Удаление стоп-слов и лемматизация
        lemmas = []
        for token in tokens:
            if token not in self.russian_stop_words and len(token) > 2:
                parsed = self.morph.parse(token)[0]
                lemmas.append(parsed.normal_form)
        
        return ' '.join(lemmas)

    def predict(self, text: str) -> dict: #полный пайплайн: предобработка + предсказание
        processed_text = self.preprocess(text)
        if not processed_text:
            return {
                "toxicity_score": 0.0
                }
        try:
            probs = self.pipeline.predict_proba([processed_text])
            return {
                "toxicity_score": int(round(probs[0][1], 2) * 100)
            }
        except:
            return {
                "toxicity_score": 812
            }

# if __name__ == "__main__":
#     model_path = "/Users/tomilovdima/good_bad_news/artefacts/toxic_model_v1.pkl"
#     russian_stop_words_path = "/Users/tomilovdima/good_bad_news/data/russian_stop_words"
#     model = ToxityModel(russian_stop_words_path, model_path)
#     test_texts = [
#     ("Ты просто глупый", "мягкое оскорбление"),
#     ("Ненавижу тебя", "сильная негативная эмоция"),
#     ("Мне не нравится твое поведение", "конструктивная критика"),
#     ("Иди отсюда", "агрессивное указание"),
#     ("Ты неправ", "несогласие без оскорблений"),
#     ("Умри, тварь", "экстремальная токсичность"),
#     ("Я разочарован", "негативная эмоция без оскорблений"),
#     ("Твоя мать...", "незавершенное оскорбление"),
#     ("Говно", "просто плохое слово"),
#     ("Не будь таким", "совет без оскорблений"),
#     ]
#     for text in test_texts:
#         print(model.predict(text[0]))
