# LegalQA: Intelligens Jogi Kérdés-Válasz Rendszer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A **LegalQA** egy fejlett, RAG (Retrieval-Augmented Generation) architektúrán alapuló rendszer, amely képes nagyméretű jogi dokumentum-adatbázisok tartalmát megérteni és azokkal kapcsolatban feltett kérdésekre kontextuális, pontos válaszokat adni.

## Főbb Funkciók

- **Dinamikus Dokumentumfeldolgozás:** A rendszer képes betölteni és feldolgozni előkészített jogi dokumentumokat.
- **Intelligens Kérdés-Válasz:** A LangChain és OpenAI modellek segítségével kontextus-érzékeny válaszokat generál.
- **Moduláris Architektúra:** A projekt logikája tiszta, karbantartható és bővíthető modulokba van szervezve.
- **Webes Felület:** Egy egyszerű, Flask alapú webes interfész a rendszerrel való interakcióhoz.
- **Visszajelzés és Elemzés:** Lehetőséget biztosít a felhasználói visszajelzések gyűjtésére és azok későbbi elemzésére.
- **Biztonságos Működés:** A kritikus műveletek, mint a titkos kulcsok kezelése és a parancsfuttatás, biztonsági szempontok figyelembevételével lettek kialakítva.

## Architektúra és Technológiai Háttér

A rendszer egy modern, Python alapú technológiai készletre épül, amelynek központi elemei a következők:

- **Backend:** Flask
- **AI/LLM:** LangChain, OpenAI
- **Adatfeldolgozás és Keresés:** Pandas, FAISS (Facebook AI Similarity Search)
- **Kódstruktúra:** A projekt logikája a `src/` könyvtárban található, moduláris felépítésben:
  - `data_processing.py`: Adatok betöltése és gyorsítótárazása (`DocumentCache`).
  - `retrieval.py`: Releváns dokumentumok visszakeresése (`CustomRetriever`).
  - `feedback.py`: Felhasználói visszajelzések kezelése (`UserFeedback`, `FeedbackAnalyzer`).
  - `learning.py`: Minőségbiztosítási és tanulási funkciók (`QualityAssurance`, `AutoLearner`).
  - `qa_system.py`: A rendszer központi vezérlő logikája (`LegalQASystem`).

## Telepítés és Beüzemelés

Az alábbi lépésekkel telepítheted és futtathatod a projektet a saját gépeden.

### 1. Kódtár klónozása
```bash
git clone https://github.com/a-te-felhasznaloneved/LegalQA.git
cd LegalQA
```

### 2. Virtuális Környezet és Függőségek
Ajánlott egy virtuális környezetet létrehozni a projekt számára.
```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
# vagy
venv\Scripts\activate  # Windows
```
Telepítsd a szükséges csomagokat:
```bash
pip install -r requirements.txt
```

### 3. Környezeti Változók
A rendszernek szüksége van egy OpenAI API kulcsra a működéshez. Hozz létre egy `.env` fájlt a projekt gyökérkönyvtárában, és add hozzá a kulcsodat:
```
OPENAI_API_KEY="a-te-openai-api-kulcsod"
```
**Fontos:** A `.gitignore` fájl be van állítva úgy, hogy ezt a fájlt ne töltse fel a kódtárba.

### 4. Adatok előkészítése
A rendszer a `processed_data/` könyvtárban keresi a feldolgozott adatokat. Mivel ezek a nagyméretű fájlok nem részei a kódtárnak, neked kell biztosítanod őket. A betöltő logika a következő fájlokat várja:
- `processed_data/processed_documents_with_embeddings.parquet`
- `processed_data/faiss_id_mapping.pkl`
- `processed_data/faiss_index.bin`
- `processed_data/graph_data/document_graph.gpickle` (opcionális)

### 5. Alkalmazás indítása
Indítsd el a Flask webalkalmazást:
```bash
python app.py
```
Az alkalmazás alapértelmezetten a `http://127.0.0.1:5001` címen lesz elérhető.

## Használat
1.  **Nyisd meg a webes felületet:** Navigálj a böngésződben a `http://127.0.0.1:5001` címre.
2.  **Dokumentumok betöltése:** Az első lépés a "Dokumentumok betöltése" gombra kattintani. Ez inicializálja a rendszert a `processed_data` könyvtárban található adatokkal.
3.  **Kérdés feltevése:** Írd be a kérdésedet a szövegmezőbe, majd kattints a "Kérdés feltétele" gombra.
4.  **Visszajelzés:** A válasz megjelenése után lehetőséged van értékelni azt és szöveges megjegyzést fűzni hozzá.

## Licenc
Ez a projekt a [MIT Licenc](LICENSE) alatt érhető el.
