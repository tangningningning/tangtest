param(
    [Parameter(Mandatory = $true)][string]$TrajectoryPath,
    [Parameter(Mandatory = $true)][int]$ExpectedSteps,
    [Parameter(Mandatory = $true)][double]$DtS,
    [Parameter(Mandatory = $true)][string]$TimelinePath,
    [Parameter(Mandatory = $true)][string]$ObservedPath
)

$ErrorActionPreference = 'Stop'

Add-Type -TypeDefinition @'
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.IO.Compression;
using System.Text;
using System.Xml;

public static class SumofcdValidator
{
    private sealed class TimeRange
    {
        public double First;
        public double Last;
        public TimeRange(double time) { First = time; Last = time; }
    }

    public static string Scan(
        string trajectoryPath,
        int expectedSteps,
        double dtS,
        string timelinePath,
        string observedPath)
    {
        if (expectedSteps <= 0 || !double.IsFinite(dtS) || dtS <= 0)
            throw new ArgumentOutOfRangeException("Expected step count and time step must be positive.");

        string timelinePartial = timelinePath + ".partial";
        string observedPartial = observedPath + ".partial";
        var observed = new Dictionary<string, TimeRange>(StringComparer.Ordinal);
        var idsAtStep = new HashSet<string>(StringComparer.Ordinal);
        long totalRows = 0;
        int timestepCount = 0;
        int currentStep = -1;
        int rowsAtStep = 0;
        bool rootSeen = false;
        bool rootClosed = false;

        try
        {
            var settings = new XmlReaderSettings
            {
                IgnoreWhitespace = true,
                DtdProcessing = DtdProcessing.Ignore,
                CheckCharacters = false,
                CloseInput = true
            };

            using (var input = File.OpenRead(trajectoryPath))
            using (var gzip = new GZipStream(input, CompressionMode.Decompress))
            using (var reader = XmlReader.Create(gzip, settings))
            using (var timeline = new StreamWriter(timelinePartial, false, new UTF8Encoding(false), 1 << 20))
            {
                while (reader.Read())
                {
                    if (reader.NodeType == XmlNodeType.Element)
                    {
                        if (reader.Name == "fcd-export")
                        {
                            if (rootSeen) throw new InvalidDataException("FCD has more than one root element.");
                            rootSeen = true;
                        }
                        else if (reader.Name == "timestep")
                        {
                            if (!rootSeen || currentStep >= 0)
                                throw new InvalidDataException("Nested or out-of-root FCD timestep.");
                            string rawTime = reader.GetAttribute("time");
                            if (rawTime == null) throw new InvalidDataException("FCD timestep has no time attribute.");
                            double timeS = double.Parse(rawTime, CultureInfo.InvariantCulture);
                            int step = checked((int)Math.Round(timeS / dtS, MidpointRounding.AwayFromZero));
                            if (Math.Abs(timeS - step * dtS) > 1e-6)
                                throw new InvalidDataException("FCD time is off-grid: " + rawTime);
                            if (step != timestepCount)
                                throw new InvalidDataException("FCD timestep missing, duplicated, or out of order at " + rawTime);
                            if (step >= expectedSteps)
                                throw new InvalidDataException("FCD contains data beyond the expected horizon at " + rawTime);
                            currentStep = step;
                            rowsAtStep = 0;
                            idsAtStep.Clear();
                            if (reader.IsEmptyElement) FinishStep();
                        }
                        else if (reader.Name == "vehicle")
                        {
                            if (currentStep < 0) throw new InvalidDataException("FCD vehicle appears outside a timestep.");
                            string vehicleId = reader.GetAttribute("id");
                            if (String.IsNullOrEmpty(vehicleId)) throw new InvalidDataException("FCD vehicle has no ID.");
                            if (!idsAtStep.Add(vehicleId))
                                throw new InvalidDataException("FCD repeats vehicle " + vehicleId + " at time " + (currentStep * dtS).ToString("R", CultureInfo.InvariantCulture));
                            double timeS = currentStep * dtS;
                            if (observed.TryGetValue(vehicleId, out TimeRange range)) range.Last = timeS;
                            else observed.Add(vehicleId, new TimeRange(timeS));
                            rowsAtStep++;
                            totalRows++;
                        }
                    }
                    else if (reader.NodeType == XmlNodeType.EndElement)
                    {
                        if (reader.Name == "timestep") FinishStep();
                        else if (reader.Name == "fcd-export") rootClosed = true;
                    }
                }

                if (currentStep >= 0 || !rootSeen || !rootClosed || timestepCount != expectedSteps)
                    throw new InvalidDataException("FCD is incomplete: steps=" + timestepCount + "/" + expectedSteps + ", open=" + currentStep + ", rootClosed=" + rootClosed);

                void FinishStep()
                {
                    if (currentStep < 0) throw new InvalidDataException("FCD timestep close has no open timestep.");
                    timeline.Write(timestepCount.ToString(CultureInfo.InvariantCulture));
                    timeline.Write('\t');
                    timeline.Write(rowsAtStep.ToString(CultureInfo.InvariantCulture));
                    timeline.Write('\n');
                    timestepCount++;
                    currentStep = -1;
                }
            }

            using (var output = new StreamWriter(observedPartial, false, new UTF8Encoding(false), 1 << 20))
            {
                foreach (var entry in observed)
                {
                    output.Write(entry.Key);
                    output.Write('\t');
                    output.Write(entry.Value.First.ToString("R", CultureInfo.InvariantCulture));
                    output.Write('\t');
                    output.Write(entry.Value.Last.ToString("R", CultureInfo.InvariantCulture));
                    output.Write('\n');
                }
            }
            File.Move(timelinePartial, timelinePath, true);
            File.Move(observedPartial, observedPath, true);
            return "{\"trajectory_timestep_count\":" + timestepCount.ToString(CultureInfo.InvariantCulture)
                + ",\"trajectory_vehicle_state_row_count\":" + totalRows.ToString(CultureInfo.InvariantCulture)
                + ",\"trajectory_unique_observed_vehicle_count\":" + observed.Count.ToString(CultureInfo.InvariantCulture) + "}";
        }
        catch
        {
            if (File.Exists(timelinePartial)) File.Delete(timelinePartial);
            if (File.Exists(observedPartial)) File.Delete(observedPartial);
            throw;
        }
    }
}
'@

$result = [SumofcdValidator]::Scan($TrajectoryPath, $ExpectedSteps, $DtS, $TimelinePath, $ObservedPath)
Write-Output $result
