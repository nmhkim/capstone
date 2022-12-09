

**********************************************************************************************************
*SKILL SPECIFICITY AND ATTITUDES TOWARDS IMMIGRATION, by Sergi Pardos-Prado and Carla Xena-Galindo       *
*                                                                                                        *
*Syntax file to reproduce 'LONGITUDINAL ANALYSIS' section, using "German_socioeconomic_panel.dta" dataset*
*                                                                                                        *
*Software used: Stata MP version 14                                                                      *
**********************************************************************************************************


***********************************************
*CODING AND RECODING VARIABLES BEFORE ANALYSIS*
***********************************************



sort pid syear /*sorting individual - year dataset*/
xtset pid syear /*setting panel data structure: years in individual panels*/

*findit plottig /*to find and install Bischoff's graph schemes for graph layouts, if not installed yet*/

*set scheme plottig /*once 'plottig' graph schemes have been installed*/


** Immigration variables **
capture drop borngermany isco88_2dg isco88_3dg nativeno tot_nativeno percentimmigr
gen borngermany = plj0010
recode borngermany (1=1) (2=0) (else=.)
* 1= yes --> native 0 = no --> immigrant
** Create ISCO variables **
gen isco88_2dg = substr(isco_str, 1, 2)
destring isco88_2dg, replace
*isco variable with 2 digit groups
gen isco88_3dg = substr(isco_str, 1, 3)
destring isco88_3dg, replace
*isco variable with 3 digit groups
*br isco88_2dg isco88_3dg
* For a definition of major groups see http://www.ilo.org/public/english/bureau/stat/isco/isco88/major.htm *
** Create immigration variables (3 digits)**
bysort isco88_3dg syear: egen nativeno = total(borngermany)
bysort isco88_3dg syear: egen tot_nativeno = count(borngermany)
lab var nativeno "total non-immigrants per year & isco group 3 digits"
** Create ISCO & immigration variables (3 digits) **
* Variable as a % of immigrants within each ISCO group (3 digits) *

capture drop percentimmigr

gen percentimmigr = ((tot_nativeno-nativeno)/tot_nativeno)*100
lab var percentimmigr "% immigrants by country/round & isco group 3 digits"

** Create immigration variables (2 digits)**
capture drop nativeno2d tot_nativeno2d percentimmigr2d
bysort isco88_2dg syear: egen nativeno2d = total(borngermany)
bysort isco88_2dg syear: egen tot_nativeno2d = count(borngermany)
lab var nativeno2d "total non-immigrants per year & isco group 2 digits"
* Variable as a % of immigrants within each ISCO group (2 digits) *
gen percentimmigr2d = ((tot_nativeno2d-nativeno2d)/tot_nativeno2d)*100
lab var percentimmigr2d "% immigrants by country/round & isco group 2 digits"

* Skill Specificity variables *

capture drop x  group length totgroup totworkforce numerator denominator iscosklevel skill1 skillsp 
capture drop skillsp
capture drop isco_str
gen x=1 if isco!=.
tostring isco, gen(isco_str)
gen group= substr(isco_str, 1,1)
gen length=length(isco_str)
replace group="0" if length==3 
bysort surveyyear group: egen totgroup = total(x) if x != .
*total N per group, wave
tab totgroup if group == "1"  & surveyyear == 2008
bysort surveyyear: egen totworkforce = total(x) if x != .
tab totworkforce if surveyyear == 2008
*total N (workforce) per wave
gen numerator = 0.08 if group == "1" 
replace numerator = 0.14 if group == "2"
replace numerator = 0.19 if group == "3"
replace numerator = 0.06 if group == "4" | group == "5" | group == "9"
replace numerator = 0.04 if group == "6"
replace numerator = 0.18 if group == "7" | group == "8" 
replace numerator = 0.003 if group == "0" 
/* eg: 'plant & machine operators and assemblers' (ISCO major group 8), contains 70 unit groups. 
 Total unit groups (for 10 major groups) is 390 --> 70/390=0.18  
 (see http://www.ilo.org/public/english/bureau/stat/isco/isco88/publ3.htm) */
gen denominator = totgroup/totworkforce
gen skillsp = (numerator/denominator)
*skillsp is average skill specialisation within major group: 'baseline' measure
//a.k.a 'absolute skill specificity'
**ILO does not assign an "ISCO skill-level" to ISCO88-1d group "1" (Legislators, senior officials, managers). 
//Iversen & Soskice assign the highest "ISCO skill-level" (4) to this group. 
//This measure is reffered to in Iversen & Soskice (APSR 2001) as "ISCO level of skills"
**ILO does not assign an "ISCO skill-level" to ISCO88 group "0" (Armed Forces).
// We could drop out observations in this category as they represent
// a very small percentage of the whole sample (N= 495)
gen iscosklevel = 4 if group == "1" | group == "2" 
replace iscosklevel = 3 if group == "3"
replace iscosklevel = 2 if group == "4" | group == "5" | group == "6" | group == "7" | group == "8"
replace iscosklevel = 1 if group == "9" 
replace iscosklevel =. if group == "0" 
gen skill1 = skillsp/iscosklevel if iscosklevel !=. & skillsp !=.
*skill1: relative to ISCO 4levels
capture drop school educ educyears skill2 skill22
*Variable that only includes school leaving degree
recode  pgsbil (-2/-1 7=.)  (6=1 "no degree") (1=2 "Low secondary") (2=3 "Middle secondary") ///
(3/4=4 "Upper secondary") (5=0 "other"), gen(school)
*Variable that also includes further education after school
capture drop educ
gen educ=school
replace educ = 5 if pgbbil01>0 & pgbbil01<7 & (school==2 | school==3 | school==1 | school==0) | pgbbil03==2
replace educ = 6 if pgbbil01>0 & pgbbil01<7 & school==4 | pgbbil03==2
replace educ = 7 if pgbbil02>0 & pgbbil02<11 | pgbbil03==3
capture lab def educlabel 0"Other" 1"No degree" 2"Low secondary" 3"Middle secondary" 4"Upper secondary" ///
5  "vocational" 6 "Upp sec + vocational" 7 "University degree"
lab val educ educlabel
* Years of education
capture drop educyears
gen educyears = pgbilzt
recode educyears -2=. -1=.
*skill2: relative to education 
capture drop skill2
gen skill2 = skillsp/educ if educ!=0

capture drop skill22
gen skill22 = skillsp/educyears

* Other data management
capture drop dv
recode immigrconcern 1=1 2/3=0 else=., gen(dv) /*concern over immigration binary*/

capture drop income
xtile income=netinclastmonth, nquantile(5) /*recodification of income in quintiles*/

capture drop age
gen age=syear-yearbirth /*age linear*/

capture drop temporary
gen temporary=timecontract==2 /*temporary contract dummy*/

capture drop mean_income
capture drop income_fixed
egen mean_income=mean(income), by(pid) /*mean income per individual for hybrid models in appendix*/
gen income_fixed=income-mean_income /*intra-individual de-meaned income for hybrid models in appendix*/

capture drop difficnewjob_dummy
recode difficnewjob 3=1 1/2=0 else=., gen(difficnewjob_dummy) /*difficulty to find a new job dummy*/

capture drop mean_difficulty
capture drop difficulty_fixed
egen mean_difficulty=mean(difficnewjob), by(pid) /*mean perception of difficulty to find job per individual for hybrid models in appendix*/
gen difficulty_fixed=difficnewjob-mean_difficulty /*intra-individual de-meaned perception of difficulty to find job for hybrid models in appendix*/

recode immigrconcern 1=3 2=2 3=1 else=., gen(concern_rec) /*concern over immigration linear*/

recode unemployed_reg 1=1 2=0 else=., gen(unemployed2) /*unemployed dummy*/

gen tenure = syear-currentjobyear /*years spent doing same job*/
egen tenure2 = mean(tenure), by(syear isco88_2dg) /*occupation-year average of years spent during same job*/

*Transition rates from t-1 to t (diagonal matrix) for each ISCO group

gen isco_alt=.
replace isco_alt=1 if isco88_2dg==11
replace isco_alt=2 if isco88_2dg==12
replace isco_alt=3 if isco88_2dg==13
replace isco_alt=4 if isco88_2dg==21
replace isco_alt=5 if isco88_2dg==22
replace isco_alt=6 if isco88_2dg==23
replace isco_alt=7 if isco88_2dg==24
replace isco_alt=8 if isco88_2dg==31
replace isco_alt=9 if isco88_2dg==32
replace isco_alt=10 if isco88_2dg==33
replace isco_alt=11 if isco88_2dg==34
replace isco_alt=12 if isco88_2dg==41
replace isco_alt=13 if isco88_2dg==42
replace isco_alt=14 if isco88_2dg==51
replace isco_alt=15 if isco88_2dg==52
replace isco_alt=16 if isco88_2dg==61
replace isco_alt=17 if isco88_2dg==71
replace isco_alt=18 if isco88_2dg==72
replace isco_alt=19 if isco88_2dg==73
replace isco_alt=20 if isco88_2dg==74
replace isco_alt=21 if isco88_2dg==81
replace isco_alt=22 if isco88_2dg==82
replace isco_alt=23 if isco88_2dg==83
replace isco_alt=24 if isco88_2dg==91
replace isco_alt=25 if isco88_2dg==92
replace isco_alt=26 if isco88_2dg==93
replace isco_alt=27 if isco88_2dg==99




sort pid syear 
xtset pid syear 

gen lag_isco88_2dg=l.isco88_2dg

capture drop transition
gen transition=.