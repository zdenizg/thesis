# Vocabulary Audit Report

## Summary

| Metric | Value |
|---|---|
| Vocabulary before filter | 529,588 |
| Vocabulary after filter | 54,058 |
| Total terms dropped | 475,530 |
| `no_below` | 5 |
| `no_above` | 0.5 |

### Dropped terms by document frequency

| Bucket | Count | Notes |
|---|---|---|
| doc_freq == 1 | 388,370 | Hapax legomena |
| doc_freq 2–4 | 87,091 | Below no_below=5 cutoff |
| doc_freq >= 5 | 69 | Dropped by no_above=0.5 rule |

## Table A — Top 50 most frequent dropped terms by corpus frequency

| term | doc_freq | corpus_freq |
|---|---|---|
| state | 2,892 | 41,025 |
| file | 3,336 | 39,854 |
| intelligence | 2,270 | 33,052 |
| security | 2,627 | 32,648 |
| report | 2,659 | 30,822 |
| agency | 2,904 | 30,573 |
| action | 2,754 | 30,344 |
| name | 2,648 | 29,703 |
| time | 2,715 | 28,400 |
| officer | 2,532 | 26,976 |
| cia | 2,180 | 26,152 |
| service | 2,401 | 25,244 |
| source | 2,298 | 25,198 |
| director | 2,511 | 24,839 |
| agent | 2,200 | 23,185 |
| activity | 2,429 | 23,174 |
| chief | 2,439 | 22,515 |
| case | 2,566 | 22,311 |
| general | 2,331 | 21,515 |
| form | 2,903 | 21,421 |
| request | 2,368 | 20,978 |
| official | 2,439 | 20,920 |
| united | 2,322 | 20,430 |
| group | 2,420 | 19,985 |
| operation | 2,114 | 19,727 |
| staff | 2,188 | 19,160 |
| government | 2,335 | 19,086 |
| member | 2,322 | 18,955 |
| contact | 2,397 | 18,892 |
| individual | 2,135 | 17,444 |
| special | 2,263 | 16,983 |
| year | 2,324 | 16,200 |
| two | 2,458 | 15,544 |
| matter | 2,041 | 15,272 |
| position | 2,120 | 15,132 |
| first | 2,387 | 14,736 |
| c. | 2,232 | 14,155 |
| foreign | 2,038 | 13,169 |
| work | 2,060 | 13,021 |
| a. | 2,274 | 12,852 |
| reference | 2,138 | 11,776 |
| person | 2,125 | 11,618 |
| day | 2,187 | 11,467 |
| comment | 2,548 | 11,292 |
| secret | 2,115 | 11,241 |
| part | 2,214 | 11,225 |
| received | 2,312 | 11,129 |
| u | 2,071 | 10,978 |
| present | 2,064 | 10,141 |
| title | 2,176 | 10,113 |

## Table B — Top 50 most frequent dropped terms with doc_freq 2–4

| term | doc_freq | corpus_freq |
|---|---|---|
| vkg | 3 | 415 |
| npd | 4 | 315 |
| moskalev | 4 | 183 |
| beheiren | 4 | 173 |
| siedel | 4 | 167 |
| knohl | 4 | 141 |
| fedoseev | 4 | 118 |
| conducted/expenses | 4 | 117 |
| observations/investigation | 4 | 117 |
| vice-intelligence | 4 | 114 |
| subject/case | 4 | 110 |
| pages_ | 3 | 104 |
| schewe | 4 | 100 |
| diko | 3 | 98 |
| amlasii | 2 | 92 |
| rothchild | 2 | 86 |
| sistone | 3 | 79 |
| piccerelli | 4 | 79 |
| hunkeler | 3 | 78 |
| amuts | 2 | 72 |
| cerveny | 2 | 70 |
| footnotes- | 3 | 68 |
| klobukar | 4 | 68 |
| bertotally | 2 | 67 |
| praxis | 3 | 66 |
| ruff | 2 | 65 |
| galbe | 2 | 63 |
| cadi | 3 | 63 |
| thadden | 4 | 61 |
| pbn | 3 | 59 |
| s.s.c.i. | 4 | 59 |
| caldevilla | 3 | 58 |
| schwickrath | 4 | 57 |
| mwaac | 4 | 57 |
| lewin | 2 | 56 |
| crabanac | 3 | 56 |
| grunewald | 2 | 55 |
| renteria | 2 | 55 |
| lacklen | 4 | 55 |
| cápita | 3 | 55 |
| photo-book | 3 | 55 |
| jmdevil | 4 | 54 |
| heiligman | 3 | 54 |
| physique | 4 | 53 |
| operations-general | 4 | 53 |
| aeboor | 4 | 52 |
| scaleti | 4 | 51 |
| collingwood | 4 | 51 |
| blahut | 4 | 49 |
| afosi | 2 | 48 |

## Table C — Random sample of 100 dropped terms with doc_freq == 1 (seed=42)

| term | doc_freq | corpus_freq |
|---|---|---|
| desfassilication | 1 | 1 |
| duve | 1 | 1 |
| classificare | 1 | 1 |
| entialgo | 1 | 2 |
| munismo | 1 | 1 |
| reacheds | 1 | 1 |
| recuperscian | 1 | 1 |
| natiɔn | 1 | 1 |
| asubui | 1 | 1 |
| maixò | 1 | 1 |
| inconscientemente | 1 | 1 |
| coligations | 1 | 1 |
| metischool | 1 | 1 |
| yoats | 1 | 1 |
| contared | 1 | 1 |
| heffins | 1 | 1 |
| bottchenko | 1 | 1 |
| l/h | 1 | 1 |
| riflessione | 1 | 1 |
| tlaltelalco | 1 | 1 |
| medelll | 1 | 1 |
| cagón | 1 | 1 |
| 'cand | 1 | 1 |
| cryptons | 1 | 1 |
| auardia | 1 | 1 |
| alafino | 1 | 1 |
| slqtied | 1 | 1 |
| diemantifera | 1 | 1 |
| maure-assistant | 1 | 1 |
| addnews | 1 | 1 |
| sɔlamente | 1 | 1 |
| nirli | 1 | 1 |
| viates | 1 | 1 |
| korelyugi | 1 | 1 |
| gesundheitsrinden | 1 | 1 |
| klinik | 1 | 1 |
| csvicus | 1 | 1 |
| goir | 1 | 1 |
| entry¨non-u.s. | 1 | 1 |
| tecall | 1 | 1 |
| arr^ | 1 | 1 |
| liempty/status | 1 | 1 |
| un-ker | 1 | 1 |
| |category | 1 | 1 |
| firrin | 1 | 1 |
| schleicher | 1 | 2 |
| homri | 1 | 1 |
| descubridor | 1 | 1 |
| quafters | 1 | 1 |
| ུག | 1 | 1 |
| foll-wing | 1 | 1 |
| mssiflostion | 1 | 1 |
| erdalisine | 1 | 1 |
| jacs | 1 | 1 |
| novelo | 1 | 1 |
| damacon | 1 | 1 |
| kasno | 1 | 1 |
| disin | 1 | 1 |
| sərdge | 1 | 1 |
| 'cont | 1 | 1 |
| displayez | 1 | 1 |
| desidare | 1 | 1 |
| wilb | 1 | 1 |
| gì | 1 | 1 |
| pioigi | 1 | 1 |
| flying-status | 1 | 1 |
| slento | 1 | 1 |
| kuchy | 1 | 1 |
| hiv-b | 1 | 1 |
| incubadoras | 1 | 1 |
| poietias | 1 | 1 |
| segueda | 1 | 1 |
| ¿performance | 1 | 1 |
| küse | 1 | 1 |
| swediate | 1 | 1 |
| collectie | 1 | 1 |
| rebater | 1 | 1 |
| moseg | 1 | 1 |
| undetectable | 1 | 1 |
| hausiamant | 1 | 1 |
| mostfog | 1 | 1 |
| sedžár | 1 | 1 |
| partof | 1 | 1 |
| housework | 1 | 1 |
| bandbox | 1 | 1 |
| sho/ksr | 1 | 1 |
| cheangre | 1 | 1 |
| ffile | 1 | 2 |
| goldwate | 1 | 1 |
| gunov | 1 | 1 |
| sdipy | 1 | 1 |
| crinion | 1 | 1 |
| soutiqued | 1 | 1 |
| iodge | 1 | 1 |
| wedx | 1 | 1 |
| ∙ment | 1 | 1 |
| sangalle | 1 | 1 |
| fubten | 1 | 1 |
| formal.qas | 1 | 1 |
| look-back | 1 | 1 |
