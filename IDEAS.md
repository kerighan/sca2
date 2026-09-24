# Laplace Attention: prochaines ablations

Hypothèses issues de la lecture du code, pas des gains de qualité mesurés.
Comparer loss à nombre de tokens égal **et** à temps égal, avec mêmes données,
seeds, budget de paramètres et validation par position dans le contexte.

## 1. Grouper les projections qui lisent le même z — testé le 2026-09-21

La fonction mathématique est conservée : concaténer les matrices de projection,
faire un GEMM puis découper ses sorties. Les arrondis peuvent changer selon le
GEMM. L'hypothèse était de réduire les lectures de z, les lancements et les
additions des contributions à dz au backward.

**Résultat : pas de gain démontré sur l'entraînement de la couche complète.**
Sur GB10, B=8, T=2048, d=1024, M=256, dv=384, dk=0, NG=1, C=128,
ff=5800, L=128, conv=4, GDN gate, triton_scan et bf16 compilé, après cinq
warmups puis 20 rounds appariés de dix itérations avec ordre alterné :

| Projections | Tête longue, fwd+bwd | Couche, fwd+bwd | Couche, fwd+bwd+AdamW |
|---|---:|---:|---:|
| Séparées (référence) | 24.983 ms | 87.491 ms | 86.150 ms |
| K/V | 24.831 ms (+1.22 %) | 87.450 ms (+0.08 %) | 87.234 ms (-0.18 %) |
| K/V/beta | 24.888 ms (+0.83 %) | 88.485 ms (-1.10 %) | 87.801 ms (-0.31 %) |
| K/V/beta, padding 64 | 24.362 ms (+2.60 %) | 87.693 ms (-0.55 %) | 86.867 ms (-0.11 %) |
| K/V/gate GDN | 24.994 ms (+0.17 %) | 88.220 ms (-0.50 %) | 88.652 ms (-1.03 %) |

Les pourcentages sont les gains de débit calculés à partir des ratios appariés,
pas du rapport des médianes. Comparer les bras dans cette même série ; ces
timings absolus ne sont pas une comparaison avec les mesures d'autres sessions.
K/V sur la couche donne +0,08 %, avec un intervalle bootstrap des rounds
approximativement [-0,36 % ; +0,38 %] : indiscernable du bruit. Le meilleur gain
sur la tête longue seule (+2,60 %) ne se retrouve pas sur la couche complète.
Le premier passage court (8 rounds de 5) donnait déjà K/V à +0,07 % sur la
couche et K/V/beta à -0,93 %. **Ne pas activer ce regroupement par défaut.**

Implémentation expérimentale dans `lapa/benchmarks/projections.py` uniquement :
les paramètres K/V/beta/gp gardent leurs noms et leur forme, donc les checkpoints
et les groupes de l'optimiseur restent identiques. La concaténation des poids
est faite à chaque appel ; son coût, les découpages, les copies de layout et
leurs gradients sont compris dans la mesure. `kvb_pad` arrondit la largeur
concaténée au multiple de 64 suivant ; `kvg` regroupe K/V et la projection GDN,
en laissant beta séparé. Le chemin normal d'entraînement reste séparé.
Cette expérience n'évalue pas un stockage permanent des poids dans un seul
paramètre ni une fusion avec les projections de la tête courte.

Validation : sorties, état et tous les gradients concordent en float64 ; les
cinq bras compilés bf16 passent aussi sur les dimensions cibles, avec une gate
beta non constante. Le benchmark porte sur une couche synthétique, pas sur
l'entraînement d'un LM complet. AdamW utilise lr=0 pour stabiliser les poids.

```bash
OMP_NUM_THREADS=1 python -m unittest lapa.test_projection_pack -v
OMP_NUM_THREADS=1 PYTORCH_ALLOC_CONF=expandable_segments:True \
  python -m lapa.benchmarks.projections \
  --packs separate,kv,kvb,kvb_pad,kvg --rounds 20 --iters 10
```

Mesures brutes et écarts de validation :
[`projection_packing_gb10_20260921.json`](lapa/benchmarks/results/projection_packing_gb10_20260921.json).

## 2. Séparer « combien écrire » et « combien effacer »

Aujourd'hui `e = W @ (v - beta * r)` : beta module l'effacement, mais pas la
valeur nouvelle. Essayer une variante `e = W @ (alpha * v - beta * r)`, avec
alpha scalaire par token. Première ablation sans nouveaux paramètres :
alpha = beta. Puis alpha appris, initialisé pour reproduire alpha = 1.
Le solve reste le même ; le supplément est un produit élémentaire fusionnable.
Mesurer norme de l'état, saturation de beta, rappel à longue distance et loss.
Ne pas activer cette modification dans le kernel de référence sans flag :
elle change réellement la règle de mémoire.

### Ablation disponible : `--beta-write`

La variante **alpha = beta** est implémentée, désactivée par défaut. Ajouter
`--beta-write` à la commande `pretrain.py --variant lapa_cc` existante
(`--variant lapa` fonctionne aussi). En Python : `LaplaceConfig(beta_write=True)`
ou `LayerCfg(..., beta_write=True)`. Le flag est transmis à la couche et enregistré
avec les arguments de l'entraînement. Aucun paramètre ni état supplémentaire.

```bash
# Ajouter à la commande d'entraînement existante, avec ses données et dimensions :
# --variant lapa_cc --beta-write

# Comparaison de vitesse isolée sur la configuration dv=384 :
OMP_NUM_THREADS=1 PYTORCH_ALLOC_CONF=expandable_segments:True \
  python -m lapa.benchmarks.triton --paths batched,triton_scan \
  --tokens 2048 --dv 384 --kv-dk 0 --groups 1 --ff 5800 \
  --short-window 128 --gdn-gate --beta-write

# Validation de la récurrence, des gradients et du chemin Triton compilé :
OMP_NUM_THREADS=1 python -m unittest lapa.test_beta_write \
  lapa.test_triton_scan.TestScan.test_beta_write -v
```

Effet précis : la tête longue calcule `e = W @ (beta*v - beta*r)` au lieu de
`e = W @ (v - beta*r)`. Le solve W reste inchangé. `v` comprend aussi les canaux
de vérification de clé lorsque kv_dk > 0. La tête courte ne change pas.
Prefill, chunks incomplets, état transporté et decode appliquent la même règle ;
les gradients de beta comprennent l'écriture et l'effacement. Le chemin
`triton_scan` reste actif : les valeurs sont multipliées en fp32 avant le scan,
via autograd, sans nouveau kernel scan spécialisé. Le surcoût n'est pas mesuré.

**Exige `--beta-groups 1`** (valeur par défaut) : une gate par mode ne définit
pas un alpha scalaire pour la valeur. Les combinaisons invalides sont rejetées.
Compatible avec `--conv-silu`, `--gdn-gate` et les groupes de lecture longs.
Conserver les autres flags et la seed pour l'ablation ; utiliser un nouveau
label/checkpoint. Le chargement des poids reste compatible, mais le flag doit
être conservé à l'inférence puisqu'il change la dynamique de mémoire.
Avec `--beta-init -2`, la nouvelle valeur est pondérée par environ 0,119 au
départ : ne pas confondre cette ablation avec celle de l'initialisation de beta.
La variante alpha appris indépendamment n'est pas implémentée.

Validation du 2026-09-21 : les quatre tests ciblés ci-dessus passent, dont la
récurrence explicite et ses gradients en float64, l'équivalence chunk/decode
avec état initial non nul, et le scan compilé bf16 B=8, D=384, C=128 sur 16
chunks plus un reste de trois tokens. Aucun gain de qualité n'est encore mesuré.

## 3. Retester le partage du budget entre têtes à dv=384

`dv` agrandit simultanément la mémoire longue, les valeurs courtes et la
projection de mélange. Mesurer le temps et l'utilité marginale de chaque tête
avant de passer à 512. Une largeur longue/courte indépendante permettrait de
donner les paramètres à la tête qui en bénéficie. Comparer à paramètres et
temps égaux, avec une ablation de tête sur checkpoint avant tout entraînement.

## 4. Calibrer l'initialisation avec conv_silu

La convolution identité suivie de SiLU n'est plus une identité. Elle change
la distribution commune aux phases, valeurs et gates. Mesurer RMS de z,
dispersion de theta*K(z), beta et contributions des branches à l'initialisation,
puis après quelques centaines de steps. Tester seulement ensuite un gain
post-SiLU ou une initialisation theta/LayerScale recalibrée. Les phases et la
delta rule étaient déjà non linéaires sans SiLU : le gain éventuel ne doit pas
être attribué à une ancienne couche « entièrement linéaire ».

## 5. Exploiter les ablations de gate déjà présentes

`gdn_gate_scope={long,both,concat,mix}` et `wiki_gate_ablation.sh` existent déjà.
Ne pas réimplémenter cette piste. Examiner leurs résultats par classe/position
avant de combiner avec conv_silu : les deux changent les amplitudes et peuvent
interagir. Garder un contrôle où seule l'une des deux options change.

## Piste plus coûteuse : oubli dépendant du token

`decay_input` et `beta_groups` existent mais quittent le chemin scan rapide.
Un benchmark qualité court peut établir leur intérêt avant de financer leur
fusion. Pour l'oubli variable, les rampes deviennent des sommes cumulées par
chunk ; il faut revoir les bornes numériques et les adjoints, pas simplement
retirer la condition de fallback.
