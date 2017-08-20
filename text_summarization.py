
from gensim.summarization import summarize
from gensim.summarization import keywords

text = "Aadhaar is essentially a paperless online anytime-anywhere identity assigned to a resident to cover his/her entire lifetime " + \
    "The verification of his identity is done online with the help of authentication devices " + \
    "which connect to UIDAI’s Central Identity Repository and return only a ‘yes’ or ‘no’ response to the basic query " + \
    "The Aadhaar authentication service is fully functional " + \
    "and in use in several service delivery schemes across the country. " + \
    "The Aadhaar Card or the e-Aadhaar (electronic copy of Aadhaar) are essentially given to residents to know their own Aadhaar" + \
    "but are only the first step towards the actual use of the online id as explained in the preceding para. " + \
    "Section 139AA of the Income-tax Act, 1961 as introduced by the Finance Act, " + \
    "2017 provides for mandatory quoting of Aadhaar / Enrolment ID of Aadhaar application form " + \
    "for filing of return of income and for making an application for  " + \
    "allotment of Permanent Account Number with effect from 1st July, 2017. " + \
    "Presently there is no Policy to give up Aadhaar. " + \
    "secure electronic document which should be treated at par with printed Aadhaar letter. " + \
    "e-adhaar "

print(text)

print(summarize(text))

print(summarize(text, split=True))

print(summarize(text,  word_count=50))  

print(keywords(text))


