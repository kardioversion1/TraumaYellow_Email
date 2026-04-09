# cerner_raw/

Raw Cerner CSV exports live here. These are the source files used to build
`data/ed_counts.csv` via manual review and the retrain pipeline.

## Naming convention

```
ED_ActivityLog_YYYY-MM-DD.csv        ED Activity Log export (one file per pull)
ED_AllPatients_YYYY-MM-DD.csv        All Patients report
DoorToTreatment_YYYY-MM-DD.csv       Door-to-Treatment times
BedOccupancy_YYYY-MM-DD.csv          Bed Occupancy report
COVID_Flu_RSV_YYYY-MM-DD.csv         Testing report
```

Date in filename = date of export (not date range of data).

## What to do after a new Cerner pull

1. Drop the raw CSV(s) here with the correct filename
2. Update `data/ed_counts.csv` with the new daily totals
3. Trigger the retrain workflow (Actions tab) or run `scripts/retrain.py` locally
4. The pipeline will update:
   - `data/predictions_history.csv`
   - `data/model_metrics.json`
   - `predictions/predictions.json`
   - `model/model.pkl`

## Outstanding data gaps

- Boarding and external transfer patient rows not yet pulled from Cerner
- Full 18-month ED Activity Log not yet pulled (currently Jul 2024 - Mar 2026)
- Only Jewish Hospital Downtown is currently in the model

## Notes

Files in this folder are committed to the private repo. They contain only
aggregate-level operational data -- no patient identifiers.
