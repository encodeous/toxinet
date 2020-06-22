using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using CsvHelper;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;

namespace ToxinetTrain
{
    class Program
    {

        static MLContext _mlContext = new MLContext();
        static void Main(string[] args)
        {
            _mlContext.Log += MlContextOnLog;
            Console.WriteLine("Loading Data...");
            using var reader = new StreamReader("Data/train.csv");
            using var csv = new CsvReader(reader, CultureInfo.InvariantCulture);
            var records = csv.GetRecords<Comment>();
            var cleanRecords = new List<CommentCleaned>();

            var clean_list = new[]
            {
                "you",
                "youre",
                "trick",
                "stick",
                "trigger",
                "ucky",
                "lucky",
                "duck",
                "nicker",
                "igg",
                "white",
                "pig",
                "chicken",
                "cow",
                "kiss",
                "balls",
                "bug",
                "assassin",
                "ask",
                "assessment",
                "methodology",
                "skin",
                "before",
                "condo",
                "go",
                "hit",
                "it",
                "u",
                "f",
                "kill",
                "stuck",
                "truck",
                "struck",
                "uck",
                "puck",
                "ur soo good",
                "do"
            };

            var dirty_list = new[]
            {
                "arse",
                "bitch",
                "bugger",
                "cunt",
                "slut",
                "murder",
                "strange",
                "harass",
                "porn",
                "meth",
                "foreskin",
                "condom",
                "chink",
                "kill",
                "kys",
                "f u c k",
                "n i g g e r",
                "s h i t"
            };

            foreach (var comment in records)
            {
                string sanitized = SanitizeString(comment.comment_text);

                int sum = comment.toxic + comment.severe_toxic + comment.obscene + comment.threat + comment.insult +
                          comment.identity_hate;

                if (sum == 0) cleanRecords.Add(new CommentCleaned { Comment = sanitized, Class = 0 });
                if (comment.toxic == 1) cleanRecords.Add(new CommentCleaned { Comment = sanitized, Class = 1 });
                if (comment.severe_toxic == 1) cleanRecords.Add(new CommentCleaned { Comment = sanitized, Class = 2 });
                if (comment.obscene == 1) cleanRecords.Add(new CommentCleaned { Comment = sanitized, Class = 3 });
                if (comment.threat == 1) cleanRecords.Add(new CommentCleaned { Comment = sanitized, Class = 4 });
                if (comment.insult == 1) cleanRecords.Add(new CommentCleaned { Comment = sanitized, Class = 5 });
                if (comment.identity_hate == 1) cleanRecords.Add(new CommentCleaned { Comment = sanitized, Class = 6 });

            }

            for (int i = 0; i < 100; i++)
            {
                foreach (string s in File.ReadAllLines("Data/clean_dictionary.txt").Union(clean_list))
                {
                    cleanRecords.Add(new CommentCleaned { Comment = s, Class = 0 });
                }

                foreach (string s in dirty_list)
                {
                    cleanRecords.Add(new CommentCleaned { Comment = s, Class = 3 });
                }
            }

            var writer = new StreamWriter("Data/train-data.csv");
            var csvw = new CsvWriter(writer, CultureInfo.InvariantCulture);
            csvw.WriteRecords(cleanRecords);
            csvw.Flush();
            writer.Flush();
            csvw.Dispose();
            writer.Dispose();

            Console.WriteLine("Processing...");

            var _trainingDataView = _mlContext.Data.LoadFromTextFile<CommentCleaned>("Data/train-data.csv", hasHeader: true, separatorChar:',');


            var pipeline =
                _mlContext.Transforms.Text.FeaturizeText("Features", nameof(CommentCleaned.Comment))
                    .Append(_mlContext.Transforms.Conversion.MapValueToKey("Label", nameof(CommentCleaned.Class)))
                    .AppendCacheCheckpoint(_mlContext)
                    .Append(_mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy(new SdcaMaximumEntropyMulticlassTrainer.Options()
                    {
                        NumberOfThreads = 2, MaximumNumberOfIterations = int.MaxValue/500
                    }))
                    .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            Console.WriteLine("Training...");

            var _trainedModel = pipeline.Fit(_trainingDataView);

            _mlContext.Model.Save(_trainedModel, _trainingDataView.Schema, "Data/model");

            var _predEngine = _mlContext.Model.CreatePredictionEngine<CommentCleaned, CommentPrediction>(_trainedModel);

            Console.WriteLine("Training Complete!");

            while (true)
            {
                string s = Console.ReadLine();
                var comment = new CommentCleaned(){Comment = SanitizeString(s)};
                Console.WriteLine();

                var prediction = _predEngine.Predict(comment);

                for(int i = 0; i < prediction.Score.Length; i++)
                {
                    Console.WriteLine($"Type: {i} Score: {prediction.Score[i] * 100}%");
                }
            }
        }

        private static void MlContextOnLog(object? sender, LoggingEventArgs e)
        {
            Console.WriteLine(e.Message);
        }

        private static Dictionary<string, float> GetScoresWithLabelsSorted(DataViewSchema schema, string name, float[] scores)
        {
            Dictionary<string, float> result = new Dictionary<string, float>();

            var column = schema.GetColumnOrNull(name);

            var slotNames = new VBuffer<ReadOnlyMemory<char>>();
            column.Value.GetSlotNames(ref slotNames);
            var names = new string[slotNames.Length];
            var num = 0;
            foreach (var denseValue in slotNames.DenseValues())
            {
                result.Add(denseValue.ToString(), scores[num++]);
            }

            return result.OrderByDescending(c => c.Value).ToDictionary(i => i.Key, i => i.Value);
        }

        public static string SanitizeString(string k)
        {
            StringBuilder sb = new StringBuilder();
            foreach (char c in k)
            {
                if (c == ' ' || char.IsLetter(c))
                {
                    sb.Append(c);
                }

                if (c == '\n')
                {
                    sb.Append(" ");
                }
            }

            return sb.ToString();
        }
    }
}
