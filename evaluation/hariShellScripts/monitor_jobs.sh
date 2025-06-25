#!/bin/bash
# Monitor parallel MedCalc-Bench jobs

echo "🔍 MedCalc-Bench Job Status"
echo "=========================="
squeue -u $USER -o "%.10i %.12j %.8T %.10M %.6D %R"

echo ""
echo "📊 Job Details:"
for job_id in 5076059 5076060 5076061 5076062; do
    echo "  Job $job_id: $(squeue -j $job_id -h -o "%T %M" 2>/dev/null || echo "COMPLETED/NOT_FOUND")"
done

echo ""
echo "📁 Output Files:"
ls -la ../outputs/*partition*.jsonl 2>/dev/null || echo "  No partition output files found yet"

echo ""
echo "🔍 Recent Log Activity:"
find ../logs -name "medcalc_p*_*.out" -newer ../logs 2>/dev/null | head -5 | while read logfile; do
    echo "  $logfile: $(tail -1 "$logfile" 2>/dev/null || echo "empty")"
done
