using System;
using System.IO;
using System.Reflection;
using System.Text;
using Microsoft.ML;

namespace toxinet
{
    public class ToxiNet
    {
        private static ToxiNet instance = null;

        private PredictionEngine<CommentCleaned, CommentPrediction> engine;

        private ToxiNet()
        {
            // Private Constructor
        }

        private void Initialize()
        {
            using var stream = Assembly.GetExecutingAssembly().GetManifestResourceStream("toxinet.model");

            var mlContext = new MLContext();
            var model = mlContext.Model.Load(stream, out var schema);
            engine = mlContext.Model.CreatePredictionEngine<CommentCleaned, CommentPrediction>(model);
        }

        public ToxiNetResult[] Predict(string comment)
        {
            string sanitized = SanitizeString(comment);

            var cleaned = new CommentCleaned(){Comment = sanitized};

            var prediction = engine.Predict(cleaned);

            return new[]
            {
                new ToxiNetResult(0, prediction.Score[0]),
                new ToxiNetResult((ToxicityType)1, prediction.Score[1]),
                new ToxiNetResult((ToxicityType)2, prediction.Score[2]),
                new ToxiNetResult((ToxicityType)3, prediction.Score[3]),
                new ToxiNetResult((ToxicityType)4, prediction.Score[4]),
                new ToxiNetResult((ToxicityType)5, prediction.Score[5]),
                new ToxiNetResult((ToxicityType)6, prediction.Score[6]),
            };
        }

        public static string SanitizeString(string k)
        {
            StringBuilder sb = new StringBuilder();
            foreach (char c in k)
            {
                if (char.IsLetter(c))
                {
                    sb.Append(c);
                }
            }

            return sb.ToString();
        }

        public static ToxiNet GetToxiNet()
        {
            if (instance == null)
            {
                instance = new ToxiNet();
                instance.Initialize();
            }

            return instance;
        }
    }
}
