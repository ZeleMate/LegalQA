# LegalQA

Jogi kérdés-válasz rendszer RAG (Retrieval-Augmented Generation) alapú megoldással.

## Leírás

A LegalQA egy olyan rendszer, amely jogi dokumentumok feldolgozására és kérdések megválaszolására szolgál. A rendszer a következő fő funkciókat tartalmazza:

- Dokumentumok betöltése és feldolgozása
- Kérdések megválaszolása a betöltött dokumentumok alapján
- Webes felület a könnyű használathoz

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

## API Endpoints

### Dokumentumok betöltése
- **URL**: `/load_documents`
- **Metódus**: `POST`
- **Válasz**: 
  ```json
  {
    "status": "success",
    "message": "Dokumentumok sikeresen betöltve!"
  }
  ```

### Kérdés megválaszolása
- **URL**: `/ask`
- **Metódus**: `POST`
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
    "answer": "A válasz szövege"
  }
  ```

## Technikai részletek

A rendszer a következő fő komponensekből áll:

1. **Flask Backend** (`app.py`)
   - REST API végpontok kezelése
   - CORS támogatás
   - Hibakezelés

2. **LegalQASystem** (`main.py`)
   - Dokumentumok feldolgozása
   - Kérdések megválaszolása
   - RAG implementáció

3. **Webes Felület** (`templates/index.html`)
   - Felhasználóbarát interfész
   - Dokumentumok betöltése
   - Kérdések megválaszolása

## Fejlesztés

A projekt további fejlesztési lehetőségei:
- Több dokumentumformátum támogatása
- Fejlettebb keresési algoritmusok
- Felhasználói fiókok és jogosultságkezelés
- Válaszok minőségének javítása
- Teljesítmény optimalizálás
