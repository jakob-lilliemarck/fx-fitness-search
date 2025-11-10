```sh
client request-optimization \
    --type-name testing \
    --fitness-goal "MIN(0.0)" \
    --schedule "GENERATIONAL(10,10)" \
    --selector "TOURNAMENT(5,20)" \
    --mutagen "MUTAGEN(0.6,0.3)" \
    --initial-population 20
```
