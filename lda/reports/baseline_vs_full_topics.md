# Baseline vs. Full Pipeline — Top-10 Words per Topic

Side-by-side top-10 words for the 25 topics produced by the full preprocessing pipeline and by the minimal-preprocessing baseline, both at k = 25 and with identical LDA hyperparameters (lda/specs/lda_params.md). Topics flagged with **(noise)** had ≥ 3 of their top-10 words in the blacklist + boilerplate noise list.

Note: topic_id alignment is positional only — there is no guarantee that topic 0 in one model corresponds to topic 0 in the other. The two columns are listed by index for convenience, not because they describe the same theme.

| topic_id | full pipeline (top-10) | baseline (top-10) |
|---------:|:-----------------------|:------------------|
| 0 | fbi, investigation, chicago, attorney, department, federal, police, rosselli, maheu, illinois | nw, york, bureau, fbi, advised, national, communist, organization, woman, jaffe |
| 1 | party, national, american, klan, communist, university, organization, program, york, committee | committee, fbi, nw, select, ssc, bureau, senate, inv, legal, material |
| 2 | cover, contract, personnel, operational, travel, month, employee, payment, required, per | black, bpp, panther, party, people, chicago, nw, illinois, police, advised |
| 3 | committee, fbi, bureau, select, ssc, material, senate, investigation, attorney, king | soviet, mexico, embassy, doc, said, wife, loginov, jfk, id, told |
| 4 | de, la, en, el, los, del, se, por, con, un | station, mexico, info, city, message, ref, cite, dispatch, project, reproduction |
| 5 | miami, cuban, call, florida, know, say, washington, cuba, york, told | line, king, enter, nw, fbi, tax, q., investigation, sa, committee |
| 6 | de, cuban, miami, cuba, info, total, ref, caracas, message, madrid | informant, bureau, index, section, must, card, investigation, organization, letter, field |
| 7 | soviet, loginov, wife, told, de, embassy, mexico, subj, asked, meet | woman, january, vietnam, congress, american, committee, war, san, francisco, mrs. |
| 8 | informant, bureau, investigation, field, letter, potential, organization, statement, department, index | de, la, que, el, en, del, total, cuba, por, para |
| 9 | line, enter, tax, schedule, total, loss, income, amount, instruction, gain | board, p., pp, public, jfk, national, cover, material, june, committee |
| 10 | committee, fbi, investigation, select, attorney, department, bureau, house, material, senate | committee, house, employee, salary, payroll, appointment, complete, representative, adjustment, authorizing |
| 11 | info, city, mexico, ray, cuba, rid, ref, cite, charles, cuban | cuban, item, hunt, board, cuba, classified, public, field, distribution, jfk |
| 12 | woman, people, january, washington, american, york, committee, party, war, congress | folder, nw, photocopy, material, gerald, library, shall, communication, check, dated |
| 13 | nosenko, kgb, soviet, american, department, moscow, embassy, ussr, second, directorate | miami, charles, nw, said, cuban, oswald, p., fbi, castro, stated |
| 14 | code, personnel, data, salary, grade, address, city, yes, signature, duty | fbi, investigation, nw, department, attorney, program, bureau, committee, country, national |
| 15 | performance, rating, duty, employee, specific, letter, supervisor, current, period, training | cuban, info, cuba, message, ray, station, bosch, caracas, miami, cable |
| 16 | fbi, bureau, york, committee, jaffe, attorney, washington, san, national, soviet | nosenko, kgb, soviet, american, department, moscow, said, embassy, top, section |
| 17 | cover, city, project, address, operational, san, york, costa, country, halperin | n't, know, top, think, said, nw, thing, say, people, get |
| 18 | index, party, organization, investigation, card, black, communist, bureau, people, panther | performance, duty, rating, section, employee, specific, supervisor, letter, period, training |
| 19 | know, think, thing, people, say, washington, get, question, senator, go | address, country, city, section, yes, birth, school, military, street, give |
| 20 | black, bpp, chicago, panther, illinois, party, people, police, hampton, detroit | personnel, code, data, station, salary, grade, mo, da, signature, employee |
| 21 | cuba, cuban, president, castro, assassination, meeting, military, soviet, kennedy, country | de, la, que, en, el, los, del, se, con, por |
| 22 | info, message, ref, city, unit, reproduction, per, issuing, letter, press | levison, ny, de, la, king, ruiz, stanley, cp, o'dell, party |
| 23 | committee, employee, house, salary, appointment, payroll, complete, representative, adjustment, authorizing | contract, cover, operational, approval, division, project, employee, dossier, central, attached |
| 24 | mexico, project, city, soviet, embassy, oswald, cuban, dispatch, headquarters, classification | cuba, cuban, castro, soviet, military, communist, country, top, political, support |
