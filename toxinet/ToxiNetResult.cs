using System;
using System.Collections.Generic;
using System.Text;

namespace toxinet
{
    public class ToxiNetResult
    {
        internal ToxiNetResult(ToxicityType type, float percent)
        {
            PredictionType = type;
            Prediction = percent;
        }

        public ToxicityType PredictionType { get; }

        public float Prediction { get; }
    }
}
