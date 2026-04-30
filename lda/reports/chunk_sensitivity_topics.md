# Chunk-Size Sensitivity — Top-10 Words per Topic

Each section reports the 25 topics from an independent LDA model trained at k = 25 with the frozen hyperparameters (lda/specs/lda_params.md), differing only in the chunk-size threshold applied during the isolated Phase 6B rerun.

## chunk_size = 3000

- num_docs: 5,203
- vocab_size: 54,927
- c_v: 0.5908

| topic | top-10 words |
|------:|:-------------|
| 0 | vkg, payment, halperin, research, project, dependent, dci, per, correspondence, proprietary |
| 1 | cuba, cuban, castro, soviet, president, meeting, u.s., american, military, support |
| 2 | cover, contract, operational, project, approval, dispatch, required, travel, headquarters, month |
| 3 | de, la, en, el, los, del, se, por, con, un |
| 4 | de, abortion, york, woman, national, wonaac, coalition, charles, swp, boston |
| 5 | chicago, illinois, bpp, black, fbi, hampton, panther, federal, bureau, investigation |
| 6 | investigation, miami, york, bureau, assigned, robert, percent, florida, counsel, attorney |
| 7 | code, personnel, data, salary, grade, designation, effective, signature, career, previous |
| 8 | people, know, say, thing, get, think, go, told, like, going |
| 9 | investigation, index, bureau, communist, card, party, subversive, program, informant, field |
| 10 | info, ref, mexico, cuban, city, cite, message, reproduction, prohibited, cuba |
| 11 | committee, fbi, select, senate, bureau, investigation, attorney, material, counsel, department |
| 12 | informant, bureau, letter, sac, field, statement, confidential, index, potential, symbol |
| 13 | fbi, john, classification, title, identification, jfk, york, control, assassination, national |
| 14 | project, mexico, gerende, board, dated, city, june, item, meeting, surveillance |
| 15 | country, york, bureau, soviet, program, coverage, department, field, establishment, informant |
| 16 | assassination, board, oswald, president, jfk, kennedy, washington, senator, think, paul |
| 17 | performance, rating, employee, duty, specific, letter, training, period, cuba, cuban |
| 18 | nosenko, kgb, soviet, embassy, american, moscow, department, ussr, second, directorate |
| 19 | duty, address, city, employee, yes, country, indicate, training, give, title |
| 20 | line, tax, enter, total, income, amount, schedule, loss, instruction, gain |
| 21 | black, party, panther, people, bpp, washington, woman, january, california, national |
| 22 | committee, employee, house, salary, complete, appointment, payroll, representative, adjustment, authorizing |
| 23 | king, march, novel, memphis, sturgis, fbi, martin, committee, luther, igrs |
| 24 | fbi, committee, attorney, bureau, ssc, king, investigation, letter, surveillance, material |

## chunk_size = 5000

- num_docs: 4,023
- vocab_size: 54,027
- c_v: 0.5978

| topic | top-10 words |
|------:|:-------------|
| 0 | told, cuban, soviet, asked, cuba, know, say, embassy, friend, get |
| 1 | fbi, committee, bureau, attorney, select, investigation, senate, ssc, material, department |
| 2 | king, committee, robert, hsca, fbi, investigation, house, attorney, assassination, martin |
| 3 | know, assassination, think, president, meeting, washington, question, thing, area, people |
| 4 | city, info, mexico, ref, cite, rid, message, reproduction, operational, total |
| 5 | investigation, bureau, index, organization, card, field, letter, law, subversive, communist |
| 6 | miami, charles, florida, haiti, counsel, researcher, duvalier, investigator, haitian, rothman |
| 7 | nosenko, kgb, american, address, department, city, country, january, school, washington |
| 8 | de, der, die, fund, und, york, total, ich, program, da |
| 9 | informant, bureau, investigation, letter, furnished, confidential, organization, york, potential, fbi |
| 10 | line, tax, enter, income, amount, total, schedule, contract, loss, payment |
| 11 | performance, rating, employee, specific, letter, duty, period, overall, supervisor, responsibility |
| 12 | angleton, committee, igrs, phone, irs, people, thing, please, scheer, internal |
| 13 | personnel, data, employee, grade, signature, code, career, duty, cover, salary |
| 14 | jaffe, soviet, york, hunt, moscow, american, kgb, embassy, bureau, fbi |
| 15 | committee, employee, house, salary, complete, appointment, representative, payroll, chairman, adjustment |
| 16 | cuba, cuban, country, president, board, soviet, program, assassination, national, communist |
| 17 | code, salary, data, personnel, classification, grade, previous, rate, step, effective |
| 18 | king, ray, fbi, committee, levison, sclc, atlanta, item, o'dell, cuba |
| 19 | duty, performance, rating, employee, supervisor, training, specific, job, responsibility, assignment |
| 20 | mexico, soviet, project, city, embassy, operational, dispatch, meeting, target, lifeat |
| 21 | committee, fbi, seidel, bpp, interview, baker, oswald, ssc, petersen, sa |
| 22 | de, la, en, el, los, del, se, por, con, un |
| 23 | black, people, party, panther, bpp, chicago, illinois, police, national, program |
| 24 | info, mexico, dir, message, ref, tichborn, issuing, reproduction, cite, cover |

## chunk_size = 10000

- num_docs: 3,156
- vocab_size: 52,844
- c_v: 0.5426

| topic | top-10 words |
|------:|:-------------|
| 0 | nosenko, kgb, soviet, american, moscow, department, embassy, agent, jaffe, ussr |
| 1 | de, la, madrid, ruiz, le, charles, say, wife, subj, know |
| 2 | performance, duty, employee, rating, position, line, specific, work, personnel, address |
| 3 | black, party, people, bpp, panther, chicago, illinois, police, program, fbi |
| 4 | de, hunt, cuba, birth, jmwave, miami, payment, country, reference, frank |
| 5 | president, assassination, think, operation, know, meeting, general, senator, washington, question |
| 6 | committee, fbi, select, senate, general, bureau, ssc, material, attorney, investigation |
| 7 | fbi, miami, investigation, assassination, kennedy, york, florida, department, agent, attorney |
| 8 | committee, employee, house, salary, appointment, payroll, complete, adjustment, authorizing, representative |
| 9 | info, woman, york, washington, ref, cite, reproduction, january, message, city |
| 10 | city, mexico, country, assassination, doc, jfk, coverage, id, rowton, nsa |
| 11 | know, well, people, thing, get, operation, american, area, cuban, think |
| 12 | committee, counsel, investigation, select, house, agent, robert, fbi, general, bureau |
| 13 | fbi, bureau, investigation, general, informant, attorney, program, department, organization, matter |
| 14 | rico, scheer, galan, halpern, committee, francisco, meeting, san, interviewer, victor |
| 15 | agent, dated, operational, approval, matter, percent, internal, per, march, june |
| 16 | project, operational, cover, operation, mexico, city, headquarters, dispatch, agent, required |
| 17 | board, assassination, jfk, release, oswald, public, material, collection, president, kennedy |
| 18 | contract, cover, agent, tichborn, operational, dossier, chin, travel, reference, desk |
| 19 | cuba, cuban, castro, operation, soviet, president, military, communist, general, meeting |
| 20 | fbi, committee, king, bureau, attorney, ssc, general, riha, communist, matter |
| 21 | mexico, embassy, soviet, city, cuban, loginov, oswald, classification, told, mexican |
| 22 | informant, code, personnel, data, salary, position, bureau, designation, grade, authority |
| 23 | de, la, en, el, los, del, se, por, con, para |
| 24 | bureau, index, investigation, individual, organization, card, field, informant, agent, letter |
