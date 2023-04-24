# Cover Type Forest
*Analiza i utworzenie modeli ML*

---

Opis wszystkich zadań znajdują sie w folderze **Reports**

Kontener uruchamiamy poleceniem

```console
docker compose up
```

Server działa pod adresem
```console
127.0.0.1:8080
```

## Opis plików
- **Reports/Analiza i heurystyka.ipynb** - wstępna analiza danych i prymitwna strategia diagnostyczna
- **Reports/Modele ML.ipynb** - opis kreacji modeli ML
- **Reports/Tensor Flow.ipynb** - opis utworzenia sieci neuronowej
- **Reports/Porównanie modeli.ipynb** - porównanie poprawności utworzonych modeli
---
- **covtype.data** - dane źródłowe
- **Reports/covtype.info** - opis danych
---
- **MyHeurstic.py** - moduł zawierający naiwne rozwiązanie predykcji
- **ModelSVC.sav** - zapisany jeden z modeli utworzony w **Modele ML.ipynb**
- **ModelLinearDiscriminant.sav** - zapisany jeden z modeli utworzony w **Modele ML.ipynb**
- **ModelNN** - sieć neuronowa utworzona w  **Tensor Flow.ipynb**
---
- **server.py** - REST API napisane w Flasku
- **templates** - html dla REST API
---
- **Dockerfile** - kontener dockerowy
- **docker-compose.yaml** - docker compose
- **env.yaml** - opis środowiska conda w kontenerze