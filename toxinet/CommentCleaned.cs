using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Data;

namespace toxinet
{
    internal class CommentCleaned
    {
        [LoadColumn(0)]
        public string Comment { get; set; }
        /// <summary>
        /// Type 0 - none
        /// 1 - toxic
        /// 2 - severe_toxic
        /// 3 - obscene
        /// 4 - threat
        /// 5 - insult
        /// 6 - identity_hate
        /// </summary>
        [LoadColumn(1)]
        public float Class { get; set; }
    }
}
