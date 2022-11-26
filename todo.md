# Opdracht 3 - Parallelisme CUDA

**De centrale vraag:**
Hoe een oplossing schaalt wanneer we meer rekeneenheden gebruiken
(schaal ~ rekeneenheden)

* onafhankelijke rekeneenheden *NVIDEA GPUs Streaming Multiprocessors (SM)s*
* controle uitoefenen -> kies # blocks
    * kies threads / block
* <s>niet 1 CUDA thread per datapunt</s>
  * benodigd # blocks BEREKENEN ~ # datapunten
* 1,2,... blocks -> cuda thread typisch meer dan 1 datapunt behandelen


## CODE
- [ ] threads per block?
- [ ] meet uitvoertijd voor ≠ # blocks
- [ ] benodigd # blocks BEREKENEN ~ # datapunten
- [ ] ``--blocks`` = hoeveel blocks gebruikt worden
- [ ] ``--threads`` = hoeveel threads per block
- [ ] Zorgt dat uw code werkt (met printf) -> <s>DEBUG HOEFT NIET</s>
- [ ] Reductie? Gebruik thrust <s># blocks of # threads</s>
- [ ] Kies datasets en parameters die passen bij je aanpak

## Metingen
- [ ] # THREADS BEPAAL ZELF 
  - [ ] # Threads **VAST**
  - [ ] veelvoud van 32!

## Collab
- [ ] zet # threads die we gebruikt hebben erin
- [ ] Dezelfde vragen als openmp

# TODO
- [ ] zet eerst je functie om in C code (zie of die werkt)
  - [ ] gebruik pointers
  - [ ] 