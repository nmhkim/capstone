
sort pid syear /*sorting individual - year dataset*/
xtset pid syear /*setting panel data structure: years in individual panels*/

*Individual mean of all skill specificity variables per individual for hybrid models in Appendix

capture drop mean_skillsp
capture drop mean_skill1
capture drop mean_skill22
capture drop mean_our
capture drop mean_tenure
capture drop mean_transition
egen mean_skillsp=mean(skillsp), by(pid)
egen mean_skill1=mean(skill1), by(pid)
egen mean_skill22=mean(skill22), by(pid)
egen mean_our=mean(our), by(pid)
egen mean_tenure=mean(tenure2), by(pid)
egen mean_transition=mean(transition), by(pid)

*Intra-individual de-meaned skill specificity variables for hybrid models in Appendix


capture drop skillsp_fixed
capture drop skill1_fixed
capture drop skill22_fixed
capture drop our_fixed
capture drop tenure_fixed
capture drop transition_fixed
gen skillsp_fixed=skillsp-mean_skillsp
gen skill1_fixed=skill1-mean_skill1
gen skill22_fixed=skill22-mean_skill22
gen our_fixed=our-mean_our
gen tenure_fixed=tenure2-mean_tenure
gen transition_fixed=transition-mean_transition


*Missing variable count to select only valid observations across individuals. ///
/// First missing count variable includes all variables except for transition, to avoid losing the first time observation ///
/// (since transition is a difference between t-1 and t). Second count variable includes the variable transition.

capture drop missing
egen missing = rmiss(percentimmigr difficnewjob_dummy skillsp skill1 skill22 income manualwork unemployed_reg gender age churchatt temporary educyears tenure2)
egen missing2 = rmiss(percentimmigr difficnewjob_dummy skillsp skill1 skill22 income manualwork unemployed_reg gender age churchatt temporary educyears tenure2 transition)



*********
*TABLE 2*
*********


xtlogit difficnewjob_dummy c.skillsp income manualwork unemployed_reg gender age churchatt temporary educyears percentimmigr if missing==0, fe
est store m1
margins, dydx(skillsp manualwork income educyears) post 
*coefplot, xline(0) levels(95) xtitle(Average Marginal Effects) coeflabel(skillsp = "Skill Specificity" income = "Income" ///
manualwork = "Working Class" educyears = "Education") ylabel(, labsize(medium)) xtitle("AMEs predicting difficulty new job", size(medium)) byopts(xrescale) name(A, replace)

xtlogit difficnewjob_dummy our income manualwork unemployed_reg gender age churchatt temporary educyears percentimmigr if missing==0, fe
est store m2
margins, dydx(our manualwork income educyears) post   
*coefplot, xline(0) levels(95) xtitle(Average Marginal Effects) coeflabel(our = "Occup Unempl" income = "Income" ///
manualwork = "Working Class" educyears = "Education") ylabel(, labsize(medium)) xtitle("AMEs predicting difficulty new job", size(medium)) byopts(xrescale) name(B, replace)

xtlogit difficnewjob_dummy tenure2 income manualwork unemployed_reg gender age churchatt temporary educyears percentimmigr if missing==0, fe
est store m3
margins, dydx(tenure2 manualwork income educyears) post
*coefplot, xline(0) levels(95) xtitle(Average Marginal Effects) coeflabel(tenure2 = "Tenure" income = "Income" ///
manualwork = "Working Class" educyears = "Education") ylabel(, labsize(medium)) xtitle("AMEs predicting difficulty new job", size(medium)) byopts(xrescale) name(C, replace)

xtlogit difficnewjob_dummy transition income manualwork unemployed_reg gender age churchatt temporary educyears percentimmigr if missing2==0, fe
est store m4
margins, dydx(transition manualwork income educyears) post
*coefplot, xline(0) levels(90) xtitle(Average Marginal Effects) coeflabel(transition = "Transition" income = "Income" ///
manualwork = "Working Class" educyears = "Education") ylabel(, labsize(medium)) xtitle("AMEs predicting difficulty new job", size(medium)) byopts(xrescale) name(D, replace)




*********
*TABLE 3*
*********


xtlogit dv difficnewjob_dummy skillsp income manualwork unemployed_reg gender age churchatt temporary educyears percentimmigr if missing==0, fe
est store m5
margins, dydx(difficnewjob_dummy manualwork income educyears) post
*coefplot, xline(0) levels(95) xtitle(Average Marginal Effects) coeflabel(difficnewjob_dummy = "Difficulty New Job" income = "Income" ///
manualwork = "Working Class" educyears = "Education") ylabel(, labsize(medium)) xtitle("AMEs predicting anti-immigrant concern", size(medium)) byopts(xrescale) name(A2, replace)


xtlogit dv difficnewjob_dummy our income manualwork unemployed_reg gender age churchatt temporary educyears percentimmigr if missing==0, fe
est store m6
margins, dydx(difficnewjob_dummy manualwork income educyears) post   
*coefplot, xline(0) levels(95) xtitle(Average Marginal Effects) coeflabel(difficnewjob_dummy = "Difficulty New Job" income = "Income" ///
manualwork = "Working Class" educyears = "Education") ylabel(, labsize(medium)) xtitle("AMEs predicting anti-immigrant concern", size(medium)) byopts(xrescale) name(B2, replace)


xtlogit dv difficnewjob_dummy tenure2 income manualwork unemployed_reg gender age churchatt temporary educyears percentimmigr if missing==0, fe
est store m7
margins, dydx(difficnewjob_dummy manualwork income educyears) post
*coefplot, xline(0) levels(95) xtitle(Average Marginal Effects) coeflabel(difficnewjob_dummy = "Difficulty New Job" income = "Income" ///
manualwork = "Working Class" educyears = "Education") ylabel(, labsize(medium)) xtitle("AMEs predicting anti-immigrant concern", size(medium)) byopts(xrescale) name(C2, replace)


xtlogit dv difficnewjob_dummy transition income manualwork unemployed_reg gender age churchatt temporary educyears percentimmigr if missing2==0, fe
est store m8
margins, dydx(difficnewjob_dummy manualwork income educyears) post
*coefplot, xline(0) levels(95) xtitle(Average Marginal Effects) coeflabel(difficnewjob_dummy = "Difficulty New Job" income = "Income" ///
manualwork = "Working Class" educyears = "Education") ylabel(, labsize(medium)) xtitle("AMEs predicting anti-immigrant concern", size(medium)) byopts(xrescale) name(D2, replace)


**********
*FIGURE 5*
**********
  
 *graph combine A A2
 
 *graph combine B B2
 
 *graph combine C C2
 
 *graph combine D D2
 

 *Mentioned in page 19 in the text: Z tests showing skill specificity coefficients significantly different across models.

dis (0.17 - 0.04) / sqrt(0.05^2 + 0.04^2) /*Z score: difference between skill specificity coefficient across model 1 and model 5*/

dis (0.09 - 0.06) / sqrt(0.01^2 + 0.01^2) /*Z score: difference between skill specificity coefficient across model 2 and model 6*/

dis (0.02 - 0.005) / sqrt(0.01^2 + 0.005^2) /*Z score: difference between tenure coefficient across model 3 and model 7*/

dis (0.28 - (-0.19)) / sqrt(0.17^2 + 0.14^2) /*Z score: difference between transition coefficient across model 4 and model 8*/




****************************
*DIFFERENCES IN DIFFERENCES*
****************************

*********
*TABLE 4*
*********

capture drop treated
gen treated=0
replace treated=1 if syear==2003 & unemployed2==1
replace treated=1 if syear==2004 & unemployed2==1
replace treated=1 if syear==2005 & unemployed2==1

xtset pid syear

xtreg concern_rec treated i.syear, fe vce(cluster pid)
est store diff1

xtreg concern_rec treated i.syear if l.skillsp>0.82, fe vce(cluster pid)
est store diff2

xtreg concern_rec treated i.syear if l.skillsp<0.82, fe vce(cluster pid)
est store diff3



*With our


xtreg concern_rec treated i.syear if l.our>6.71, fe vce(cluster pid)
est store diff4

xtreg concern_rec treated i.syear if l.our<6.71, fe vce(cluster pid)
est store diff5

*With tenure

xtreg concern_rec treated i.syear if l.tenure2>11.6, fe vce(cluster pid)
est store diff6

xtreg concern_rec treated i.syear if l.tenure2<11.6, fe vce(cluster pid)
est store diff7


*With transition


xtreg concern_rec treated i.syear if l.transition>0.71, fe vce(cluster pid)
est store diff8

xtreg concern_rec treated i.syear if l.transition<0.71, fe vce(cluster pid)
est store diff9




************************************************************************************************************************
*UPPER PANEL TABLE 5 (EXPLORING MECHANISMS): unemployed at t-1 more likely to find a mini/midi job during Hartz reforms*
************************************************************************************************************************

tab plb0187

capture drop minijob
recode plb0187 1/2=1 3=0 else=., gen(minijob)

capture drop hartz
gen hartz=0
replace hartz=1 if syear==2003
replace hartz=1 if syear==2004
replace hartz=1 if syear==2005

capture drop unemp_hartz
gen unemp_hartz=0
replace unemp_hartz=1 if l.unemployed2==1 & hartz==1


xtreg minijob unemp_hartz i.syear, fe vce(cluster pid)
est store mj1
xtreg minijob unemp_hartz i.syear if l.skillsp>0.82, fe vce(cluster pid)
est store mj2
xtreg minijob unemp_hartz i.syear if l.skillsp<0.82, fe vce(cluster pid)
est store mj3
xtreg minijob unemp_hartz i.syear if l.our>6.71, fe vce(cluster pid)
est store mj4
xtreg minijob unemp_hartz i.syear if l.our<6.71, fe vce(cluster pid)
est store mj5
xtreg minijob unemp_hartz i.syear if l.tenure2>11.6, fe vce(cluster pid)
est store mj6
xtreg minijob unemp_hartz i.syear if l.tenure2<11.6, fe vce(cluster pid)
est store mj7
xtreg minijob unemp_hartz i.syear if l.transition>0.71, fe vce(cluster pid)
est store mj8
xtreg minijob unemp_hartz i.syear if l.transition<0.71, fe vce(cluster pid)
est store mj9


****************************************************************************************************************************************************
*LOWER PANEL TABLE 5 (EXPLORING MECHANISMS): unemployed at t-1 who got a minijob during Hartz reforms are less likely to worry about getting as job*
****************************************************************************************************************************************************

  
capture drop mini_unemp_hartz
gen mini_unemp_hartz=0
replace mini_unemp_hartz =1 if minijob==1 & l.unemployed2==1 & hartz==1


xtreg difficnewjob mini_unemp_hartz i.syear, fe vce(cluster pid)
est store mj10
xtreg difficnewjob mini_unemp_hartz i.syear if l.skillsp>0.82, fe vce(cluster pid)
est store mj11
xtreg difficnewjob mini_unemp_hartz i.syear if l.skillsp<0.82, fe vce(cluster pid)
est store mj12
xtreg difficnewjob mini_unemp_hartz i.syear if l.our>6.71, fe vce(cluster pid)
est store mj13
xtreg difficnewjob mini_unemp_hartz i.syear if l.our<6.71, fe vce(cluster pid)
est store mj14
xtreg difficnewjob mini_unemp_hartz i.syear if l.tenure2>11.6, fe vce(cluster pid)
est store mj15
xtreg difficnewjob mini_unemp_hartz i.syear if l.tenure2<11.6, fe vce(cluster pid)
est store mj16
xtreg difficnewjob mini_unemp_hartz i.syear if l.transition>0.71, fe vce(cluster pid)
est store mj17
xtreg difficnewjob mini_unemp_hartz i.syear if l.transition<0.71, fe vce(cluster pid)
est store mj18

  
*MENTIONED IN PAGE 24 IN THE TEXT: Abadie's semi-parametric diff-in-diff estimator. Control group weihted by propensity to be selected into treatment.
  

findit absdid /*find and installed absdid package if not installed yet*/  
  
capture drop treated
gen treated=0
replace treated=1 if syear==2003 & unemployed2==1
replace treated=1 if syear==2004 & unemployed2==1
replace treated=1 if syear==2005 & unemployed2==1

  
absdid concern_rec, tvar(treated) xvar(income manualwork gender age temporary educyears skillsp our)

*Save resulting data as "gsp_data_final.dta" or "gsp_data_final.csv"