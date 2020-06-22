using System;
using System.Collections.Generic;
using System.Text;

namespace ToxinetTrain
{
    class Comment
    {
        public string id { get; set; }
        public string comment_text { get; set; }
        public int toxic { get; set; }
        public int severe_toxic { get; set; }
        public int obscene { get; set; }
        public int threat { get; set; }
        public int insult { get; set; }
        public int identity_hate { get; set; }

    }
}
