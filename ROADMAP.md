# Roadmap – IVUS ANALYSIS

## Phase 1: Load in Data and bring in usable format
- [x] Load global data
- [x] Load individual pressure and ivus data combine in class object
- [] Fix bugs in Loess data -> not really necessary since Rust aligned IVUS more reliable anyways
- [] For patient data also get lumen area systole and lumen area diastole (non normalized) and save to local_patient_stats (so can calculate percent changes if wanted)

## Phase 2: Global statistics
- [x] Plot pressure change FFR, PdPa, systolic ratio  (Change FFR, iFR)
- [x] Plot lumen change, maximal lumen narrowing change
- [x] Table rest versus stress
- [x] Table iFR>0.8 versus iFR<=0.8 and FFR>0.8 versus FFR<=0.8
- [] Correlate pressure change with lumen change
- [x] All IVUS measurements as predictors for FFR<=0.8
- [x] All IVUS measurements as predictors for iFR<=0.8
- [] KDE plot area/elliptic ratio for hemodynamic relevant and non-relevant

## Phase 3: Patient statistics
- [] Quantify dynamics for pulsatile/stress-induced lumen changes
- [] Where most dynamics?
- [] Pulsatile as predictor for FFR/iFR <=0.8
- [] Pulsatile in combination with anatomical for prediction
- [] Vessels with more dynamic more often relevant?
- [] Automatically identified MLA better diagnostic performance?
- [] Check that stiffness is only summarized over the intramural course (different behaviour then normal vessel)

## Phase 4: Advanced Visualizations
- [x] Load in png with deformation information from Rust program, summarize pixels to degree groups bin into 20 percent of intramural get the values of all patients summarize in a heatmap
- [x] Add subplot to deformation map with box plots/violing plots for area below per bin 
- [] Video of a point moving over a pressure curve in realtime.