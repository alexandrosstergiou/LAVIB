
lavib00="https://www.dropbox.com/scl/fo/r3w08y0p4x23nzyvpfz0y/ADK0fEcI1RbXiI6imnV8OIU/lavib00?rlkey=k3nvqkraalxifv3547dnz30bn&dl=0"
lavib01="https://www.dropbox.com/scl/fo/r3w08y0p4x23nzyvpfz0y/APhwa8ph967Rl0S8CFEjKLU/lavib01?rlkey=k3nvqkraalxifv3547dnz30bn&dl=0"
lavib02="https://www.dropbox.com/scl/fo/r3w08y0p4x23nzyvpfz0y/ADupizBbyxaqxLw_euMSyZU/lavib02?rlkey=k3nvqkraalxifv3547dnz30bn&dl=0"
lavib03="https://www.dropbox.com/scl/fo/r3w08y0p4x23nzyvpfz0y/ANTLkzwl1JvuJHPFh49IEHs/lavib03?rlkey=k3nvqkraalxifv3547dnz30bn&dl=0"
lavib04="https://www.dropbox.com/scl/fo/r3w08y0p4x23nzyvpfz0y/AN2TJa8EdzF7kGzOKohStp0/lavib04?rlkey=k3nvqkraalxifv3547dnz30bn&dl=0"
lavib05="https://www.dropbox.com/scl/fo/r3w08y0p4x23nzyvpfz0y/ADe6eJrgkRQ1yA5KB1JGMZE/lavib05?rlkey=k3nvqkraalxifv3547dnz30bn&dl=0"
lavib06="https://www.dropbox.com/scl/fo/r3w08y0p4x23nzyvpfz0y/APiFXtw5M7g8BAN2ECglF0A/lavib06?rlkey=k3nvqkraalxifv3547dnz30bn&dl=0"
lavib07="https://www.dropbox.com/scl/fo/r3w08y0p4x23nzyvpfz0y/AL6jiEtA0N_Xv-B8mUReYdg/lavib07?rlkey=k3nvqkraalxifv3547dnz30bn&dl=0"
lavib08="https://www.dropbox.com/scl/fo/r3w08y0p4x23nzyvpfz0y/AIpUy0YLo9dekjsE5smts2Q/lavib08?rlkey=k3nvqkraalxifv3547dnz30bn&dl=0"
lavib09="https://www.dropbox.com/scl/fo/r3w08y0p4x23nzyvpfz0y/AIPODROpq9tNLvUzjcWGaYM/lavib09?rlkey=k3nvqkraalxifv3547dnz30bn&dl=0"
lavib10="https://www.dropbox.com/scl/fo/r3w08y0p4x23nzyvpfz0y/ALoDTzjYx0Cgg4P3oooFpaE/lavib10?rlkey=k3nvqkraalxifv3547dnz30bn&dl=0"

mkdir -p data

wget -O data/annotations.zip https://www.dropbox.com/scl/fo/r3w08y0p4x23nzyvpfz0y/AFNxpnjd5y8GM2CDFujk95s/annotations?rlkey=k3nvqkraalxifv3547dnz30bn&dl=0

wget -O data/lavib00 $lavib00
wget -O data/lavib01 $lavib01
wget -O data/lavib02 $lavib02
wget -O data/lavib03 $lavib03
wget -O data/lavib04 $lavib04
wget -O data/lavib05 $lavib05
wget -O data/lavib06 $lavib06
wget -O data/lavib07 $lavib07
wget -O data/lavib08 $lavib08
wget -O data/lavib09 $lavib09
wget -O data/lavib10 $lavib10

cat data/lavib* | tar xzpvf -

