"""Extract public RedisJSON RelationshipProfile values from a pinned RDB.

This is a deliberately narrow parser: accept only module-v2 keys with the
observed RedisJSON module ID, one string payload, and module EOF. Verify the
full public dump hash, entry key/header, decoded JSON primary key and length.
No Redis server, model service, or dialogue extraction is involved.
"""
import argparse, hashlib, json, re
from pathlib import Path
EXPECTED_SHA256='324c037f4ac3ca6605c800bfe74eb1c11b05ed8c96c26df48c7560a7dac90da2'
PREFIX=b':sotopia.database.persistent_profile.RelationshipProfile:'
MODULE_ID=0x45e25238df912c03

def read_length(data,pos):
    first=data[pos];pos+=1;kind=first>>6
    if kind==0:return first&63,pos,False
    if kind==1:return ((first&63)<<8)|data[pos],pos+1,False
    if first==0x80:return int.from_bytes(data[pos:pos+4],'big'),pos+4,False
    if first==0x81:return int.from_bytes(data[pos:pos+8],'big'),pos+8,False
    if kind==3:return first&63,pos,True
    raise ValueError('unsupported RDB length')

def lzf_decompress(raw,expected_length):
    out=bytearray();i=0
    while i<len(raw):
        ctrl=raw[i];i+=1
        if ctrl<32:
            n=ctrl+1;out.extend(raw[i:i+n]);i+=n
        else:
            n=ctrl>>5;offset=(ctrl&31)<<8
            if n==7:n+=raw[i];i+=1
            offset+=raw[i];i+=1;ref=len(out)-offset-1;n+=2
            if ref<0:raise ValueError('bad LZF back-reference')
            for _ in range(n):out.append(out[ref]);ref+=1
    if len(out)!=expected_length:raise ValueError('LZF length mismatch')
    return bytes(out)

def read_string(data,pos):
    length,pos,encoded=read_length(data,pos)
    if not encoded:return data[pos:pos+length],pos+length
    if length!=3:raise ValueError('expected ordinary or LZF string')
    compressed,pos,e=read_length(data,pos);assert not e
    expanded,pos,e=read_length(data,pos);assert not e
    return lzf_decompress(data[pos:pos+compressed],expanded),pos+compressed

def extract(path, profile_type='RelationshipProfile'):
    if profile_type not in ['RelationshipProfile','EnvironmentProfile','AgentProfile']:raise ValueError('unsupported profile type')
    prefix=(':sotopia.database.persistent_profile.'+profile_type+':').encode()
    data=Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest()!=EXPECTED_SHA256:raise ValueError('public dump SHA256 mismatch')
    if data[:9]!=b'REDIS0011':raise ValueError('unexpected RDB version')
    rows=[];receipts=[]
    for match in re.finditer(re.escape(prefix)+rb'01[A-Z0-9]{24}',data):
        start,end=match.span();key=data[start:end]
        # All profile keys have 14-bit length encoding and module-v2 type 7.
        if data[start-3]!=7:raise ValueError('not a module-v2 key')
        length,pos,encoded=read_length(data,start-2)
        if encoded or pos!=start or length!=len(key):raise ValueError('key length mismatch')
        module,pos,encoded=read_length(data,end)
        if encoded or module!=MODULE_ID:raise ValueError('unexpected RedisJSON module')
        opcode,pos,encoded=read_length(data,pos)
        if encoded or opcode!=5:raise ValueError('expected module string opcode')
        raw,pos=read_string(data,pos)
        if data[pos]!=0:raise ValueError('expected module EOF')
        row=json.loads(raw)
        if row.get('pk')!=key[len(prefix):].decode():raise ValueError('profile key mismatch')
        required={'RelationshipProfile':['agent_1_id','agent_2_id','relationship','background_story'],'EnvironmentProfile':['scenario','agent_goals','relationship'],'AgentProfile':['first_name','last_name','age','occupation','secret']}[profile_type]
        if not all(k in row for k in required):raise ValueError('incomplete '+profile_type)
        rows.append(row);receipts.append({'pk':row['pk'],'rdb_key_offset':start,'json_sha256':hashlib.sha256(raw).hexdigest()})
    if not rows:raise ValueError('no RelationshipProfiles found')
    return {'source_dump_sha256':EXPECTED_SHA256,'source_revision':'e583406958ff132f6749ca87a2f9aa31ae3c0fa1','profile_type':profile_type,'relationship_profiles' if profile_type=='RelationshipProfile' else 'profiles':sorted(rows,key=lambda r:r['pk']),'extraction_receipts':receipts}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dump');p.add_argument('output');p.add_argument('--profile-type',choices=['RelationshipProfile','AgentProfile','EnvironmentProfile'],default='RelationshipProfile');a=p.parse_args();result=extract(a.dump,a.profile_type);Path(a.output).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print('Verified',a.profile_type,len(result.get('relationship_profiles',result.get('profiles',[]))))
