CONJUNTO DE DADOS PADERBORN

Artigo Marcos Rômulo – Induscon 2023

Arquivos selecionados para a implementação do sistema de diagnóstico.

Classe 01 – K001, K003, K005 e K006 – Normal.

Classe 02 – KA04 – Fadiga na pista externa – 1 – Single point

Classe 03 – KA15 – Deformação na pista externa – 1 – Single point

Classe 04 – KI04 – Fadiga na pista interna – 1 – Single point

Classe 05 – KB23 – Fadiga nas pistas externa e interna – 2 – Single point

Classe 06 - KB27 – Deformação nas pistas externa e interna – 1 - distribuído

Classe 07 – KA16 – Fadiga na pista externa – 2 – single point

Classe 08 – KB24 – Fadiga nas pistas interna e externa – 3 - distribuído

Classe 09 – KI16 – Fadiga na pista interna – 3 – Single point

Classe 10 – KI18 – Fadiga na pista interna – 2 – Single point


Linha de comando para acessar uma posição da struct com os dados do Paderborn.

```vibration = N09_M07_F10_K001_1.Y(7).Data;   % Acessa o sinal de vibração.```

```time = N09_M07_F10_K001_1.X(2).Data;   % Acessa o sinal de tempo.```