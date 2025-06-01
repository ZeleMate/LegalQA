# LegalQA

Jogi kérdés-válasz rendszer RAG (Retrieval-Augmented Generation) alapú megoldással.

## Leírás

A LegalQA egy olyan rendszer, amely jogi dokumentumok feldolgozására és kérdések megválaszolására szolgál. A rendszer a következő fő funkciókat tartalmazza:

- Dokumentumok betöltése és feldolgozása
- Kérdések megválaszolása a betöltött dokumentumok alapján
- Webes felület a könnyű használathoz
- Felhasználói visszajelzések kezelése és elemzése
- Automatikus tanulás a visszajelzések alapján
- Válaszok minőségének biztosítása és validálása
- Dokumentumok gyorsítótárazása
- Időbeli trendek és kérdési mintázatok elemzése
- Válaszok minőségének automatikus javítása

## Rendszerarchitektúra

### Fő komponensek

1. **LegalQASystem**: A rendszer fő osztálya, amely koordinálja az összes komponenst
2. **CustomRetriever**: Egyedi dokumentum kereső a jogi dokumentumokhoz
3. **QualityAssurance**: Válaszok minőségének biztosítása és validálása
4. **UserFeedback**: Felhasználói visszajelzések kezelése
5. **FeedbackAnalyzer**: Részletes visszajelzés-elemző
6. **AutoLearner**: Automatikus tanulás a visszajelzések alapján
7. **DocumentCache**: Dokumentumok gyorsítótárazása

### Adatbázis

A rendszer SQLite adatbázist használ a visszajelzések tárolására:

```sql
CREATE TABLE feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    feedback_id TEXT NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    rating INTEGER NOT NULL,
    comments TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

## Telepítés

1. Klónozd le a repository-t:
```bash
git clone [repository-url]
cd LegalQA
```

2. Telepítsd a szükséges függőségeket:
```bash
pip install -r requirements.txt
```

## Használat

1. Indítsd el a szervert:
```bash
python app.py
```

2. Nyisd meg a böngészőben: `http://localhost:5001`

3. A webes felületen keresztül:
   - Először töltsd be a dokumentumokat a "Dokumentumok betöltése" gombbal
   - Ezután tehetsz fel kérdéseket a rendszernek
   - Értékeld a válaszokat és adj visszajelzést

## API Dokumentáció

### Alap URL
```
http://localhost:5001
```

### Endpoints

#### 1. Dokumentumok betöltése
- **URL**: `/load_documents`
- **Metódus**: `POST`
- **Leírás**: Betölti és feldolgozza a jogi dokumentumokat
- **Válasz**: 
  ```json
  {
    "status": "success",
    "message": "Dokumentumok sikeresen betöltve!"
  }
  ```
- **Hibakezelés**:
  ```json
  {
    "status": "error",
    "message": "Hiba történt a dokumentumok betöltése során!"
  }
  ```

#### 2. Kérdés megválaszolása
- **URL**: `/ask`
- **Metódus**: `POST`
- **Leírás**: Megválaszol egy jogi kérdést a betöltött dokumentumok alapján
- **Kérés body**:
  ```json
  {
    "question": "A kérdés szövege"
  }
  ```
- **Válasz**:
  ```json
  {
    "status": "success",
    "answer": "A válasz szövege",
    "feedback_id": "egyedi_azonosító"
  }
  ```
- **Hibakezelés**:
  ```json
  {
    "status": "error",
    "message": "Hiba történt a kérdés megválaszolása során!"
  }
  ```

#### 3. Visszajelzés küldése
- **URL**: `/feedback`
- **Metódus**: `POST`
- **Leírás**: Felhasználói visszajelzés mentése
- **Kérés body**:
  ```json
  {
    "feedback_id": "egyedi_azonosító",
    "rating": 5,
    "comments": "Opcionális megjegyzések"
  }
  ```
- **Válasz**:
  ```json
  {
    "status": "success",
    "message": "Visszajelzés sikeresen mentve!"
  }
  ```
- **Hibakezelés**:
  ```json
  {
    "status": "error",
    "message": "Hiányzó visszajelzési adatok!"
  }
  ```

#### 4. Visszajelzések elemzése
- **URL**: `/feedback/analysis`
- **Metódus**: `GET`
- **Leírás**: Visszajelzések elemzésének lekérdezése
- **Válasz**:
  ```json
  {
    "status": "success",
    "analysis": {
      "temporal_trends": {
        "trend_irány": "javuló/romló/stabil",
        "átlagos_értékelés_változás": 0.5,
        "visszajelzések_száma_változás": 10,
        "jelentős_változások": ["változás1", "változás2"],
        "javaslatok": ["javaslat1", "javaslat2"]
      },
      "question_patterns": {
        "gyakori_kérdéstípusok": ["típus1", "típus2"],
        "legmagasabb_értékelésű_kérdések": ["kérdés1", "kérdés2"],
        "legalacsonyabb_értékelésű_kérdések": ["kérdés1", "kérdés2"],
        "javaslatok": ["javaslat1", "javaslat2"]
      },
      "answer_quality": {
        "átlagos_válasz_minőség": 4.5,
        "gyakori_hibák": ["hiba1", "hiba2"],
        "erősségek": ["erősség1", "erősség2"],
        "javulási_területek": ["terület1", "terület2"],
        "javaslatok": ["javaslat1", "javaslat2"]
      }
    }
  }
  ```
- **Hibakezelés**:
  ```json
  {
    "status": "error",
    "message": "Hiba történt az elemzés során!"
  }
  ```

## Fejlesztés

A projekt további fejlesztési lehetőségei:
- Több dokumentumformátum támogatása
- Fejlettebb keresési algoritmusok
- Felhasználói fiókok és jogosultságkezelés
- Válaszok minőségének javítása
- Teljesítmény optimalizálás
- Több nyelvű támogatás
- API dokumentáció készítése
- Automatikus tanulás a visszajelzések alapján
- Részletesebb elemzési funkciók
- Adatbázis bevezetése a visszajelzések tárolására
- Graf alapú dokumentum kapcsolatok kezelése
- Válaszok minőségének automatikus javítása
- Időbeli trendek és kérdési mintázatok elemzése
- Dokumentumok gyorsítótárazása
