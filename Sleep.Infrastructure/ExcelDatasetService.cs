using OfficeOpenXml;
using Sleep.Domain;

namespace Sleep.Infrastructure;

public class ExcelDatasetService
{
    private readonly string _filePath;
    private int _processedCount = 0;

    public ExcelDatasetService(string filePath = "sleep-dataset.xlsx")
    {
        _filePath = filePath;
        ExcelPackage.LicenseContext = LicenseContext.NonCommercial;
        LoadDataset();
    }

    public int ProcessedCount => _processedCount;

    public void LoadDataset()
    {
        if (!File.Exists(_filePath))
        {
            CreateInitialDataset();
            return;
        }

        using var package = new ExcelPackage(new FileInfo(_filePath));
        var worksheet = package.Workbook.Worksheets[0];
        
        if (worksheet == null)
        {
            CreateInitialDataset();
            return;
        }

        var rowCount = worksheet.Dimension?.Rows ?? 0;
        _processedCount = Math.Max(0, rowCount - 1);
    }

    public void AppendRecord(SleepRecord record, int qualityLabel)
    {
        FileInfo fileInfo = new FileInfo(_filePath);
        
        if (!fileInfo.Exists)
        {
            CreateInitialDataset();
            fileInfo = new FileInfo(_filePath);
        }

        using var package = new ExcelPackage(fileInfo);
        var worksheet = package.Workbook.Worksheets[0];
        
        if (worksheet == null)
        {
            worksheet = package.Workbook.Worksheets.Add("SleepData");
            WriteHeader(worksheet);
        }

        var nextRow = worksheet.Dimension?.End.Row + 1 ?? 2;
        
        worksheet.Cells[nextRow, 1].Value = record.Age;
        worksheet.Cells[nextRow, 2].Value = record.Gender;
        worksheet.Cells[nextRow, 3].Value = record.DailySteps;
        worksheet.Cells[nextRow, 4].Value = record.PhysicalActivity;
        worksheet.Cells[nextRow, 5].Value = record.StressLevel;
        worksheet.Cells[nextRow, 6].Value = record.SleepDuration;
        worksheet.Cells[nextRow, 7].Value = qualityLabel;

        package.Save();
        _processedCount++;
    }

    private void CreateInitialDataset()
    {
        using var package = new ExcelPackage();
        var worksheet = package.Workbook.Worksheets.Add("SleepData");
        
        WriteHeader(worksheet);
        
        package.SaveAs(new FileInfo(_filePath));
        _processedCount = 0;
    }

    private void WriteHeader(ExcelWorksheet worksheet)
    {
        worksheet.Cells[1, 1].Value = "Age";
        worksheet.Cells[1, 2].Value = "Gender";
        worksheet.Cells[1, 3].Value = "DailySteps";
        worksheet.Cells[1, 4].Value = "PhysicalActivity";
        worksheet.Cells[1, 5].Value = "StressLevel";
        worksheet.Cells[1, 6].Value = "SleepDuration";
        worksheet.Cells[1, 7].Value = "QualityOfSleep";
    }
}

