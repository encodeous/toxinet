using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Data;

namespace toxinet
{
    internal class CommentPrediction
    {
        [ColumnName("PredictedLabel")] 
        public float Class { get; set; }

        public float[] Score { get; set; }
    }
}
