# cerner_raw/

Raw Cerner CSV exports live here. Commit them here before running the retrain
so there is a permanent record of the source data behind every model version.

## File naming convention

```
ED_ActivityLog_YYYY-MM-DD.csv        ED Activity Log (primary visit-level export)
ED_AllPatients_YYYY-MM-DD.csv        All Patients report
DoorToTreatment_YYYY-MM-DD.csv       Door-to-Treatment times
BedOccupancy_YYYY-MM-DD.csv          Bed Occupancy / boarding report
COVID_Flu_RSV_YYYY-MM-DD.csv         COVID / Flu / RSV testing report
```

Date in filename = date of export, not the date range of data inside.

## After a new Cerner pull

1. Export the relevant reports from Cerner and save them here with the correct filename
2. Update `data/ed_counts.csv` with the new daily visit totals (one row per date)
3. Go to the Actions tab and trigger **Weekly model retrain** manually
4. The retrain pipeline will update:
   - `data/predictions_history.csv` (appends new predictions, overwrites any existing rows for those dates)
   - `data/model_metrics.json`
   - `predictions/predictions.json`
   - `model/model.pkl`

## Known gaps (as of Apr 2026)

- Boarding and external transfer patient rows not yet included in ed_counts.csv
- Full 18-month ED Activity Log not yet pulled — current coverage is Jul 2024 through Mar 2026
- Only Jewish Hospital Downtown is in the model

## Notes

This is a private repo. Files here contain aggregate operational data only -- no patient identifiers,
no dates of birth, no MRN or encounter numbers. Raw files are kept for reproducibility and audit trail.
